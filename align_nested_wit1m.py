"""
Compute nested mutual kNN alignment between DINO and LLM features.

Query:   WIT-1024 (1024 samples)
Gallery: nested subsets of WIT-1M (10K, 50K, 100K, 500K, 1M)

Each smaller gallery is a strict subset of the larger one.  Pre-computed
gallery indices (gallery_indices_*.npy, positions 0..999999 into the 1M
shard set) are loaded from --gallery_indices_dir.  These same indices
apply to both DINO and LLM shards.

Supports both original and gemini captions (only affects LLM features;
DINO features are the same regardless of caption type).

Usage:
    python align_nested_wit1m.py \
        --dino_layer 10 --llm_layer 23 \
        --query_dino_path ./results/features/wit_1024/vit_base_patch14_dinov2.lvd142m_pool-cls.pt \
        --query_llm_path  ./results/features/wit_1024/openlm-research_open_llama_3b_pool-avg.pt \
        --gallery_dino_shard_dir ./results/features/wit_1m/shards/vit_base_patch14_dinov2.lvd142m \
        --gallery_llm_shard_dir  ./results/features/wit_1m/shards/openlm-research_open_llama_3b \
        --gallery_indices_dir ./data/wit1m/nested_wit1m_indices

Other galleries (e.g. LAION-15M) only need their shard dirs, their
gallery_indices dir, --sizes and --name. The Faiss index lives in RAM: one size
costs n x (768 + 3200) x 4 bytes, ~16 GB per 1M rows (240 GB at 15M), and CPU
search time grows with n, so beyond ~1M use knn_sets.py + layer_sweep.py (GPU,
streaming, same feature preparation and identical neighbour sets).
"""

import gc
import os
import argparse
import glob

import torch
import numpy as np
from tqdm.auto import tqdm

from measure_alignment import prepare_features

# Nested gallery sizes (each is a subset of the next larger one)
NESTED_SIZES = [10_000, 50_000, 100_000, 500_000, 1_000_000]


def _normalize_l2(x):
    """In-place L2 normalization of rows (numpy array)."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    x /= norms


def load_nested_indices(indices_dir, sizes):
    """
    Load pre-computed gallery indices from .npy files.
    Indices are positions 0..999999 into the 1M shard set.

    Returns dict: {size: np.array of indices}
    """
    nested = {}
    for size in sizes:
        path = os.path.join(indices_dir, f"gallery_indices_{size}.npy")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping size {size:,}")
            continue
        indices = np.load(path)
        nested[size] = indices
        print(f"  Loaded {size:,} indices from {os.path.basename(path)} "
              f"(range [{indices.min()}, {indices.max()}])")
    return nested


def compute_nested_indices(total, sizes, seed=42):
    """
    Compute nested index arrays: each smaller set is a strict subset of
    the next larger one.  Fallback when no pre-computed indices are available.
    """
    rng = np.random.RandomState(seed)
    all_indices = rng.permutation(total)

    nested = {}
    for size in sorted(sizes, reverse=True):
        if size > total:
            print(f"  Skipping size {size:,} > total {total:,}")
            continue
        nested[size] = np.sort(all_indices[:size])

    return nested


# ---- Shard loading ----

_shard_info_cache = {}


def _get_shard_info(shard_dir):
    if shard_dir in _shard_info_cache:
        return _shard_info_cache[shard_dir]

    shard_files = sorted(glob.glob(os.path.join(shard_dir, "shard_*.pt")))
    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {shard_dir}")

    first = torch.load(shard_files[0], map_location="cpu", mmap=True)
    num_layers = first["feats"].shape[1]
    feat_dim = first["feats"].shape[2]
    layer_indices = first.get("layer_indices", None)
    del first

    total_samples = 0
    shard_sizes = []
    for sf in shard_files:
        data = torch.load(sf, map_location="cpu", mmap=True)
        shard_sizes.append(data["feats"].shape[0])
        total_samples += data["feats"].shape[0]
        del data

    info = (shard_files, shard_sizes, total_samples, num_layers, feat_dim, layer_indices)
    _shard_info_cache[shard_dir] = info
    print(f"  Shards: {len(shard_files)} files, {total_samples:,} samples, "
          f"dim={feat_dim}, layers={layer_indices or list(range(num_layers))}")
    return info


def _compute_quantile(shard_dir, layer_idx, indices, q=0.95,
                      sample_size=200_000, seed=42):
    if q >= 1.0:
        return float('inf')

    shard_files, shard_sizes, total, _, _, layer_indices = _get_shard_info(shard_dir)
    local_layer_idx = layer_idx
    if layer_indices is not None:
        local_layer_idx = layer_indices.index(layer_idx)

    cumulative = np.cumsum([0] + shard_sizes)
    n_effective = len(indices) if indices is not None else total

    rng = np.random.RandomState(seed + 99999)
    n_sample = min(sample_size, n_effective)

    if indices is not None:
        sample_pos = np.sort(rng.choice(len(indices), size=n_sample, replace=False))
        sample_global = indices[sample_pos]
    else:
        sample_global = np.sort(rng.choice(total, size=n_sample, replace=False))

    samples = []
    for shard_idx, sf in enumerate(shard_files):
        shard_start = cumulative[shard_idx]
        shard_end = cumulative[shard_idx + 1]
        mask = (sample_global >= shard_start) & (sample_global < shard_end)
        if not mask.any():
            continue
        local_idx = sample_global[mask] - shard_start
        data = torch.load(sf, map_location="cpu", mmap=True)
        chunk = data["feats"][local_idx, local_layer_idx, :].float()
        samples.append(chunk)
        del data

    all_samples = torch.cat(samples, dim=0)
    del samples
    k = int(q * all_samples.numel())
    if k >= all_samples.numel():
        k = all_samples.numel() - 1
    q_val = torch.kthvalue(all_samples.reshape(-1).abs(), k + 1).values.item()
    del all_samples
    return q_val


def _build_faiss_index(shard_dir, layer_idx, q_val, indices):
    import faiss

    shard_files, shard_sizes, _, _, feat_dim, layer_indices = _get_shard_info(shard_dir)
    local_layer_idx = layer_idx
    if layer_indices is not None:
        local_layer_idx = layer_indices.index(layer_idx)

    cumulative = np.cumsum([0] + shard_sizes)
    index = faiss.IndexFlatL2(feat_dim)

    n_added = 0
    for shard_idx, sf in enumerate(tqdm(shard_files, desc="Building index", leave=False)):
        shard_start = cumulative[shard_idx]
        shard_end = cumulative[shard_idx + 1]

        data = torch.load(sf, map_location="cpu", mmap=True)
        chunk = data["feats"][:, local_layer_idx, :].float()
        del data

        if indices is not None:
            mask = (indices >= shard_start) & (indices < shard_end)
            if not mask.any():
                del chunk
                continue
            local_idx = indices[mask] - shard_start
            chunk = chunk[local_idx]

        chunk.clamp_(-q_val, q_val)
        chunk_np = np.ascontiguousarray(chunk.numpy(), dtype=np.float32)
        del chunk

        _normalize_l2(chunk_np)
        index.add(chunk_np)
        n_added += chunk_np.shape[0]
        del chunk_np

    print(f"  Index built: {n_added:,} vectors, dim={feat_dim}")
    return index


def build_gallery_index(shard_dir, layer_idx, indices, seed=42):
    print(f"  Computing quantile...")
    q_val = _compute_quantile(shard_dir, layer_idx, indices, seed=seed)
    print(f"  q_val={q_val:.6f}")
    print(f"  Building FAISS index ({len(indices):,} samples)...")
    index = _build_faiss_index(shard_dir, layer_idx, q_val, indices)
    return index


def load_query_features(path, layer_idx, q=0.95):
    data = torch.load(path, map_location="cpu")
    feats = data["feats"].float()
    print(f"  Raw shape: {feats.shape}")
    del data

    feats = feats[:, layer_idx, :]
    feats = prepare_features(feats, q=q, exact=True)

    feats_np = np.ascontiguousarray(feats.numpy(), dtype=np.float32)
    del feats
    _normalize_l2(feats_np)
    return feats_np


def mutual_knn(query_dino_np, query_llm_np,
               gallery_dino_index, gallery_llm_index,
               topk=10, query_gallery_offset=None):
    n_query = query_dino_np.shape[0]
    search_k = topk + 1 if query_gallery_offset is not None else topk

    print(f"  Querying DINO index ({n_query} queries, search_k={search_k})...")
    dist_dino, knn_dino = gallery_dino_index.search(query_dino_np, search_k)

    print(f"  Querying LLM index ({n_query} queries, search_k={search_k})...")
    dist_llm, knn_llm = gallery_llm_index.search(query_llm_np, search_k)

    if query_gallery_offset is not None:
        clean_dino = np.zeros((n_query, topk), dtype=np.int64)
        clean_llm = np.zeros((n_query, topk), dtype=np.int64)
        clean_dist_dino = np.zeros((n_query, topk), dtype=np.float32)
        clean_dist_llm = np.zeros((n_query, topk), dtype=np.float32)
        for i in range(n_query):
            self_idx = query_gallery_offset + i
            mask_d = knn_dino[i] != self_idx
            valid_idx = knn_dino[i][mask_d][:topk]
            valid_dist = dist_dino[i][mask_d][:topk]
            clean_dino[i, :len(valid_idx)] = valid_idx
            clean_dist_dino[i, :len(valid_dist)] = valid_dist
            mask_l = knn_llm[i] != self_idx
            valid_idx = knn_llm[i][mask_l][:topk]
            valid_dist = dist_llm[i][mask_l][:topk]
            clean_llm[i, :len(valid_idx)] = valid_idx
            clean_dist_llm[i, :len(valid_dist)] = valid_dist
        knn_dino = clean_dino
        knn_llm = clean_llm
        dist_dino = clean_dist_dino
        dist_llm = clean_dist_llm

    sim_dino = 1.0 - dist_dino / 2.0
    sim_llm = 1.0 - dist_llm / 2.0

    # Per-sample intersection: broadcast for small k (fast), set intersection
    # for large k (the [n_query, k, k] broadcast OOMs when k > ~1000).
    if topk <= 1000:
        knn_dino_t = torch.from_numpy(knn_dino)
        knn_llm_t = torch.from_numpy(knn_llm)
        matches = (knn_dino_t.unsqueeze(2) == knn_llm_t.unsqueeze(1))
        per_sample_scores = matches.any(dim=2).sum(dim=1).float() / topk
    else:
        scores = np.empty(n_query, dtype=np.float32)
        for i in range(n_query):
            scores[i] = len(
                set(knn_dino[i].tolist()) & set(knn_llm[i].tolist())
            ) / topk
        per_sample_scores = torch.from_numpy(scores)

    score = per_sample_scores.mean().item()
    return score, per_sample_scores.numpy(), knn_dino, knn_llm, sim_dino, sim_llm


def run_nested_alignment(query_dino, query_llm,
                         gallery_dino_shard_dir, gallery_llm_shard_dir,
                         dino_layer, llm_layer,
                         nested_indices, seed=42):
    """
    Run alignment at each nested gallery size.
    Same indices are used for both DINO and LLM shards (both are 1M).
    """
    results = {}

    for gallery_size in sorted(nested_indices.keys()):
        indices = nested_indices[gallery_size]
        print(f"\n{'=' * 80}")
        print(f"Gallery size: {gallery_size:,}")
        print(f"{'=' * 80}")

        print(f"\nBuilding DINO gallery index...")
        dino_index = build_gallery_index(
            gallery_dino_shard_dir, dino_layer, indices, seed
        )
        print(f"\nBuilding LLM gallery index...")
        llm_index = build_gallery_index(
            gallery_llm_shard_dir, llm_layer, indices, seed
        )

        # Add query to gallery for self-matching
        dino_index.add(query_dino.copy())
        llm_index.add(query_llm.copy())
        query_gallery_offset = gallery_size
        print(f"  Gallery+query size: {dino_index.ntotal:,}")

        # k=10
        print(f"  Computing mutual kNN (k=10)...")
        score_k10, per_sample_k10, knn_dino_k10, knn_llm_k10, sim_dino_k10, sim_llm_k10 = mutual_knn(
            query_dino, query_llm, dino_index, llm_index,
            topk=10, query_gallery_offset=query_gallery_offset,
        )
        print(f"  [k=10] score={score_k10:.4f}")

        # k=1
        print(f"  Computing mutual kNN (k=1)...")
        score_k1, per_sample_k1, knn_dino_k1, knn_llm_k1, sim_dino_k1, sim_llm_k1 = mutual_knn(
            query_dino, query_llm, dino_index, llm_index,
            topk=1, query_gallery_offset=query_gallery_offset,
        )
        print(f"  [k=1] score={score_k1:.4f}")

        # k=n/100
        k_pct = max(1, gallery_size // 100)
        print(f"  Computing mutual kNN (k={k_pct})...")
        score_k_pct, per_sample_k_pct, knn_dino_k_pct, knn_llm_k_pct, sim_dino_k_pct, sim_llm_k_pct = mutual_knn(
            query_dino, query_llm, dino_index, llm_index,
            topk=k_pct, query_gallery_offset=query_gallery_offset,
        )
        print(f"  [k={k_pct}] score={score_k_pct:.4f}")

        results[gallery_size] = {
            "score_k10": score_k10,
            "per_sample_scores_k10": per_sample_k10,
            "knn_dino_k10": knn_dino_k10,
            "knn_llm_k10": knn_llm_k10,
            "sim_dino_k10": sim_dino_k10,
            "sim_llm_k10": sim_llm_k10,
            "score_k1": score_k1,
            "per_sample_scores_k1": per_sample_k1,
            "knn_dino_k1": knn_dino_k1,
            "knn_llm_k1": knn_llm_k1,
            "sim_dino_k1": sim_dino_k1,
            "sim_llm_k1": sim_llm_k1,
            "score_k_pct": score_k_pct,
            "k_pct": k_pct,
            "per_sample_scores_k_pct": per_sample_k_pct,
            "knn_dino_k_pct": knn_dino_k_pct,
            "knn_llm_k_pct": knn_llm_k_pct,
            "gallery_indices": indices,
            "gallery_size": gallery_size,
        }

        del dino_index, llm_index
        gc.collect()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino_layer", type=int, default=10)
    parser.add_argument("--llm_layer", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--caption_type", type=str, default="original",
                        choices=["original", "gemini"])
    parser.add_argument("--output_dir", type=str,
                        default="./results/alignment/nested_wit1m")
    parser.add_argument("--query_dino_path", type=str, required=True)
    parser.add_argument("--query_llm_path", type=str, required=True)
    parser.add_argument("--gallery_dino_shard_dir", type=str, required=True)
    parser.add_argument("--gallery_llm_shard_dir", type=str, required=True)
    parser.add_argument("--gallery_indices_dir", type=str, default=None,
                        help="Directory with gallery_indices_*.npy files "
                             "(positions 0..999999 into 1M shards). "
                             "If not provided, generates random nested indices.")
    parser.add_argument("--sizes", nargs="*", type=int, default=None,
                        help="Gallery sizes to evaluate (default: 10K 50K 100K 500K 1M)")
    parser.add_argument("--name", default="nested_wit1m",
                        help="output file prefix (e.g. nested_laion15m)")
    args = parser.parse_args()

    sizes = args.sizes or NESTED_SIZES

    print("=" * 80)
    print(f"Nested alignment ({args.name}): wit_1024 queries")
    print(f"DINO layer {args.dino_layer}, LLM layer {args.llm_layer}, seed {args.seed}")
    print(f"Caption type: {args.caption_type}")
    print(f"Gallery sizes: {[f'{s:,}' for s in sizes]}")
    print(f"Query DINO: {args.query_dino_path}")
    print(f"Query LLM:  {args.query_llm_path}")
    print(f"Gallery DINO: {args.gallery_dino_shard_dir}")
    print(f"Gallery LLM:  {args.gallery_llm_shard_dir}")
    print(f"Gallery indices: {args.gallery_indices_dir or '(random)'}")
    print("=" * 80)

    # Get shard info
    print(f"\nDINO shards:")
    _, _, total_dino, _, _, _ = _get_shard_info(args.gallery_dino_shard_dir)
    print(f"\nLLM shards:")
    _, _, total_llm, _, _, _ = _get_shard_info(args.gallery_llm_shard_dir)

    # Build nested indices
    if args.gallery_indices_dir is not None:
        print(f"\nLoading pre-computed gallery indices...")
        nested_indices = load_nested_indices(args.gallery_indices_dir, sizes)
    else:
        print(f"\nGenerating random nested indices...")
        nested_indices = compute_nested_indices(total_dino, sizes, seed=args.seed)

    # Load query features
    print(f"\nLoading query DINO features (layer {args.dino_layer})...")
    query_dino = load_query_features(args.query_dino_path, args.dino_layer)
    print(f"  Query DINO shape: {query_dino.shape}")

    print(f"\nLoading query LLM features (layer {args.llm_layer})...")
    query_llm = load_query_features(args.query_llm_path, args.llm_layer)
    print(f"  Query LLM shape: {query_llm.shape}")

    # Run nested alignment
    results = run_nested_alignment(
        query_dino, query_llm,
        args.gallery_dino_shard_dir, args.gallery_llm_shard_dir,
        args.dino_layer, args.llm_layer,
        nested_indices, seed=args.seed,
    )

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"Results summary:")
    for size in sorted(results.keys()):
        r = results[size]
        print(f"  {size:>10,}: k10={r['score_k10']:.4f}  "
              f"k1={r['score_k1']:.4f}  "
              f"k{r['k_pct']}={r['score_k_pct']:.4f}")
    print(f"{'=' * 80}")

    # Save: include size(s) in filename so parallel SLURM tasks don't collide
    os.makedirs(args.output_dir, exist_ok=True)
    llm_model_name = os.path.basename(args.gallery_llm_shard_dir)
    dino_model_name = os.path.basename(args.gallery_dino_shard_dir)
    caption_tag = f"_{args.caption_type}" if args.caption_type != "original" else ""

    # Size tag: e.g. "_size10000" for a single size, "_all" for the full set
    size_list = sorted(results.keys())
    if len(size_list) == 1:
        size_tag = f"_size{size_list[0]}"
    else:
        size_tag = "_all"

    save_name = (
        f"{args.name}{caption_tag}{size_tag}"
        f"_dino-L{args.dino_layer}_{dino_model_name}"
        f"_llm-L{args.llm_layer}_{llm_model_name}.npy"
    )
    save_path = os.path.join(args.output_dir, save_name)

    np.save(save_path, {
        "results": results,
        "sizes": sorted(results.keys()),
        "dino_layer": args.dino_layer,
        "llm_layer": args.llm_layer,
        "caption_type": args.caption_type,
        "seed": args.seed,
        "n_query": query_dino.shape[0],
    })
    print(f"\nSaved to {save_path}")
