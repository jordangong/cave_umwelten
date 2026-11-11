"""
Pick the (vision, language) layer pair that aligns best on WIT-1024.

The README refers to this script, but upstream never shipped it. This
reimplementation follows the PRH protocol (Huh et al., 2024) that the paper
cites for layer selection (Sec. 3.2.1, App. A.2): mutual kNN with k=10 over the
1024 query samples (gallery = the queries, self excluded), maximised over all
layer pairs. Features are prepared exactly as align_nested_wit1m.py prepares
the queries: pick the layer, clamp at the exact 95th percentile of |x|, then
l2-normalise.

Writes `--output` (JSON) with the best pair and its WIT-1024 scores at
k=1, k=10 and k=n/100, and `<output stem>_grid.npy` with the full k=10 grid.

    python find_best_layers.py \
        --vision_path   results/features/wit_1024/vit_base_patch14_dinov2.lvd142m_pool-cls.pt \
        --language_path results/features/wit_1024/openlm-research_open_llama_3b_pool-avg.pt \
        --output results/best_layers.json
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from measure_alignment import prepare_features


def load_feats(path):
    data = torch.load(path, map_location="cpu")
    return data["feats"].float(), data.get("layer_indices")


def knn_masks(feats, kmax, q, device):
    """Per layer: [N, kmax] neighbour indices (descending cosine, self excluded)."""
    out = []
    for l in range(feats.shape[1]):
        x = prepare_features(feats[:, l, :], q=q, exact=True)
        x = F.normalize(x.to(device), p=2, dim=-1)
        sim = x @ x.T
        sim.fill_diagonal_(float("-inf"))
        out.append(sim.topk(kmax, dim=1).indices)
    return out


def mutual_knn(knn_a, knn_b, k):
    """Mean fraction of shared neighbours among the top-k of each space."""
    n = knn_a.shape[0]
    a = torch.zeros(n, n, device=knn_a.device)
    b = torch.zeros(n, n, device=knn_b.device)
    a.scatter_(1, knn_a[:, :k], 1.0)
    b.scatter_(1, knn_b[:, :k], 1.0)
    return ((a * b).sum(1) / k).mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vision_path", required=True)
    p.add_argument("--language_path", required=True)
    p.add_argument("--topk", type=int, default=10, help="k used to pick the pair")
    p.add_argument("--q", type=float, default=0.95, help="outlier clamp quantile")
    p.add_argument("--output", default="results/best_layers.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vis, vis_layers = load_feats(args.vision_path)
    lang, lang_layers = load_feats(args.language_path)
    assert vis.shape[0] == lang.shape[0], (vis.shape, lang.shape)
    n = vis.shape[0]
    vis_layers = vis_layers or list(range(vis.shape[1]))
    lang_layers = lang_layers or list(range(lang.shape[1]))
    print(f"vision {tuple(vis.shape)}  language {tuple(lang.shape)}  n={n}")

    k_pct = max(1, n // 100)
    kmax = max(args.topk, 1, k_pct)
    knn_v = knn_masks(vis, kmax, args.q, device)
    knn_l = knn_masks(lang, kmax, args.q, device)

    grid = np.zeros((len(knn_v), len(knn_l)))
    for i, kv in enumerate(knn_v):
        for j, kl in enumerate(knn_l):
            grid[i, j] = mutual_knn(kv, kl, args.topk)

    bi, bj = np.unravel_index(grid.argmax(), grid.shape)
    scores = {f"k{k}": mutual_knn(knn_v[bi], knn_l[bj], k) for k in sorted({1, 10, k_pct})}
    best = {
        "vision_layer": int(vis_layers[bi]),
        "language_layer": int(lang_layers[bj]),
        "selected_with_k": args.topk,
        "n": n,
        "k_pct": k_pct,
        "scores_wit1024": scores,
        "vision_path": args.vision_path,
        "language_path": args.language_path,
    }

    print("\nk=%d mutual kNN, rows = vision layer, cols = language layer" % args.topk)
    with np.printoptions(precision=3, linewidth=250, suppress=True):
        print(grid)
    print(f"\nbest: vision layer {best['vision_layer']}, language layer {best['language_layer']}")
    print("WIT-1024 scores at best pair: " + "  ".join(f"{k}={v:.4f}" for k, v in scores.items()))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(best, f, indent=2)
    np.save(os.path.splitext(args.output)[0] + "_grid.npy", grid)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
