"""
Nested-gallery kNN sets of the 1024 WIT queries, for every layer of one model.

For each gallery size n in {1024 (queries only), 10K, 50K, 100K, 500K, 1M} and
each layer, this stores the ids of each query's top-kmax neighbours (kmax =
max(10, n // 100)), most similar first, self excluded. Mutual kNN at k=1, 10
and n/100 between ANY two models (any layer pair) then needs only these sets:
see `layer_sweep.py`. This is what App. A.2 (per-scale layer re-selection), and
A.1 / A.4 (other model pairs), need without rebuilding a Faiss index per pair.

Features are prepared exactly as `align_nested_wit1m.py` prepares them:
- queries: clamp at the exact 95th percentile of |x| over the 1024 x D layer
  slice (`prepare_features(exact=True)`), then l2-normalise;
- gallery of size n: clamp at the 95th percentile of |x| over a 200K-row sample
  of that gallery (same RandomState(42 + 99999) draw as `_compute_quantile`;
  all rows when n <= 200K), then l2-normalise;
- the queries are added to every gallery and each query's own row is dropped.
Ids: gallery rows are their global position 0..N-1 in the shard set (N = rows
in all shards: 1,000,000 for WIT-1M), query j is N + j. Exact search on GPU in
fp32 (TF32 off).

Memory is bounded by one shard plus the top-k state (L x 1024 x kmax per size,
~1.2 GB per layer at kmax = 150K), so the same code runs LAION-15M galleries
(queries are still WIT-1024 and can't be in LAION, but they're added to the
gallery the same way):

    python knn_sets.py --query_path results/features/wit_1024/<model>_pool-*.pt \
        --shard_dir results/features/laion_15m/shards/<model> \
        --gallery_indices_dir $PLATO_DATA_ROOT/jordangong--laion-15m/nested_indices \
        --sizes 10000 50000 100000 500000 1000000 5000000 10000000 15000000 \
        --output results/knn/laion_15m/<model>.pt

    python knn_sets.py \
        --query_path results/features/wit_1024/openlm-research_open_llama_3b_pool-avg.pt \
        --shard_dir  results/features/wit_1m/shards/openlm-research_open_llama_3b \
        --output     results/knn/openlm-research_open_llama_3b.pt
"""
import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

SIZES = [10_000, 50_000, 100_000, 500_000, 1_000_000]
Q = 0.95
SAMPLE = 200_000
SEED = 42


def kmax_for(n):
    return max(10, n // 100)


# ------------------------------------------------------------------ quantiles

class AbsQuantile:
    """Exact sorted(|x|)[int(q * numel)] over rows streamed in chunks.
    bf16: a histogram over the 32768 non-negative bf16 codes (bf16 ordering of
    |x| equals integer ordering of its bits), so no samples are kept.
    Other dtypes: rows are kept and torch.kthvalue runs at the end."""

    def __init__(self, num_layers):
        self.L = num_layers
        self.counts = None
        self.rows = []
        self.numel = 0

    def add(self, x):  # x: [m, L, D]
        if x.numel() == 0:
            return
        self.numel += x[:, 0].numel()
        if x.dtype == torch.bfloat16:
            codes = (x.contiguous().view(torch.int16).to(torch.int32) & 0x7FFF)
            if self.counts is None:
                self.counts = torch.zeros(self.L, 0x8000, dtype=torch.int64)
            for l in range(self.L):
                self.counts[l] += torch.bincount(codes[:, l].reshape(-1), minlength=0x8000)
        else:
            self.rows.append(x.float())

    def values(self, q=Q):
        k = int(q * self.numel)  # 0-based rank, as kthvalue(k + 1) in align_nested_wit1m
        out = []
        if self.counts is not None:
            for l in range(self.L):
                c = int(torch.searchsorted(self.counts[l].cumsum(0), k, right=True))
                out.append(torch.tensor([c], dtype=torch.int16).view(torch.bfloat16).float().item())
        else:
            x = torch.cat(self.rows)
            for l in range(self.L):
                out.append(torch.kthvalue(x[:, l].reshape(-1).abs(), k + 1).values.item())
        return out


def quantile_sample(indices, seed=SEED, sample_size=SAMPLE):
    """Global row ids of the sample _compute_quantile draws for this gallery."""
    rng = np.random.RandomState(seed + 99999)
    n_sample = min(sample_size, len(indices))
    pos = np.sort(rng.choice(len(indices), size=n_sample, replace=False))
    return indices[pos]


# ----------------------------------------------------------------- top-k state

class TopK:
    """Running top-kmax per layer. Columns are buffered until there are
    max(kmax, merge_cols) of them, then merged with one topk: at kmax = 150K
    (LAION-15M) a topk per 10k-row shard would dominate the run. The buffer
    is at most the size of the state itself."""

    def __init__(self, num_layers, kmax, n_query, device, merge_cols=0):
        self.k = kmax
        self.sim = torch.full((num_layers, n_query, kmax), float("-inf"), device=device)
        self.ids = torch.full((num_layers, n_query, kmax), -1, dtype=torch.int32, device=device)
        self.cap = max(kmax, merge_cols)
        self.buf = [([], []) for _ in range(num_layers)]
        self.nbuf = [0] * num_layers

    def merge(self, l, sim, ids):  # sim [nq, m], ids [m] int32
        self.buf[l][0].append(sim)
        self.buf[l][1].append(ids)
        self.nbuf[l] += sim.shape[1]
        if self.nbuf[l] >= self.cap:
            self._flush(l)

    def _flush(self, l):
        sims, idss = self.buf[l]
        if not sims:
            return
        s = torch.cat([self.sim[l]] + sims, dim=1)
        i = torch.cat([self.ids[l]] + [x.unsqueeze(0).expand(s.shape[0], -1) for x in idss], dim=1)
        self.buf[l], self.nbuf[l] = ([], []), 0
        top = s.topk(self.k, dim=1, sorted=False)
        del s
        self.sim[l] = top.values
        self.ids[l] = i.gather(1, top.indices)

    def finish(self):
        for l in range(len(self.buf)):
            self._flush(l)
        order = self.sim.argsort(dim=2, descending=True)
        return self.ids.gather(2, order).cpu(), self.sim.gather(2, order).cpu()


# ------------------------------------------------------------------------ main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--query_path", required=True)
    p.add_argument("--shard_dir", required=True)
    p.add_argument("--gallery_indices_dir", default="data/wit1m/nested_wit1m_indices")
    p.add_argument("--output", required=True)
    p.add_argument("--sizes", nargs="*", type=int, default=SIZES)
    p.add_argument("--layers", nargs="*", type=int, default=None,
                   help="layer ids to use (default: every layer the shards store)")
    p.add_argument("--merge_cols", type=int, default=0,
                   help="buffer gallery columns per size until max(kmax, this) before a top-k merge "
                        "(same result; fewer large-k topk calls)")
    p.add_argument("--n_total", type=int, default=None,
                   help="rows in the shard set; default: the shards' total (checked if given)")
    p.add_argument("--sample", type=int, default=SAMPLE, help="clamp-quantile sample size (tests)")
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"
    t0 = time.time()

    shard_files = sorted(glob.glob(os.path.join(args.shard_dir, "shard_*.pt")))
    assert shard_files, args.shard_dir
    sizes_n = []
    for sf in shard_files:
        sizes_n.append(torch.load(sf, map_location="cpu", mmap=True)["feats"].shape[0])
    first = torch.load(shard_files[0], map_location="cpu", mmap=True)
    shard_ids = first.get("layer_indices") or list(range(first["feats"].shape[1]))
    del first
    starts = np.cumsum([0] + sizes_n)
    N_TOTAL = QUERY_ID0 = int(starts[-1])
    assert args.n_total is None or args.n_total == N_TOTAL, (args.n_total, N_TOTAL)
    print(f"{len(shard_files)} shards, {N_TOTAL:,} gallery rows")

    sizes = sorted(args.sizes)
    nested = {n: np.load(os.path.join(args.gallery_indices_dir, f"gallery_indices_{n}.npy")) for n in sizes}
    # Membership rank: smallest size index whose gallery has this row (nested sets).
    rank = np.full(N_TOTAL, len(sizes), dtype=np.int8)
    for si in reversed(range(len(sizes))):
        rank[nested[sizes[si]]] = si
    for si, n in enumerate(sizes):
        assert (rank <= si).sum() == n, f"galleries are not nested at {n}"

    # ---- queries
    qd = torch.load(args.query_path, map_location="cpu")
    qfeats = qd["feats"]
    query_ids = qd.get("layer_indices") or list(range(qfeats.shape[1]))
    layer_ids = args.layers if args.layers is not None else shard_ids
    q_pos = [query_ids.index(i) for i in layer_ids]   # query layer axis may hold more layers
    layers = [shard_ids.index(i) for i in layer_ids]  # positions in the shards' layer axis
    L = len(layers)
    nq = qfeats.shape[0]
    qn, q_query = [], []
    for l in q_pos:
        x = qfeats[:, l, :].float()
        qv = x.reshape(-1).abs().sort().values[int(Q * x.numel())].item()
        q_query.append(qv)
        qn.append(F.normalize(x.clamp(-qv, qv), dim=-1))
    qn = torch.stack(qn).to(dev)  # [L, nq, D]
    print(f"queries {tuple(qfeats.shape)}, layer ids {layer_ids}, {len(shard_files)} shards")

    # ---- pass 1: per-size, per-layer gallery clamp values
    samples = {n: quantile_sample(nested[n], sample_size=args.sample) for n in sizes}
    qacc = {n: AbsQuantile(L) for n in sizes}
    for si, sf in enumerate(shard_files):
        lo, hi = starts[si], starts[si + 1]
        local = {n: s[(s >= lo) & (s < hi)] - lo for n, s in samples.items()}
        union = np.unique(np.concatenate(list(local.values())))
        if len(union) == 0:
            continue
        feats = torch.load(sf, map_location="cpu", mmap=True)["feats"]
        rows = feats[torch.from_numpy(union)][:, layers]
        for n in sizes:
            qacc[n].add(rows[torch.from_numpy(np.searchsorted(union, local[n]))])
        del feats, rows
    q_gal = {n: qacc[n].values() for n in sizes}
    del qacc
    print(f"pass 1 (clamp values) done, {time.time() - t0:.0f}s")
    for n in sizes:
        print(f"  n={n:>8,}  q[{layer_ids[0]}]={q_gal[n][0]:.5f}  q[{layer_ids[-1]}]={q_gal[n][-1]:.5f}")

    # ---- pass 2: exact top-k against each gallery
    state = {n: TopK(L, kmax_for(n), nq, dev, args.merge_cols) for n in sizes}
    qids = torch.arange(nq, dtype=torch.int32, device=dev) + QUERY_ID0
    eye = torch.eye(nq, dtype=torch.bool, device=dev)
    q_only = TopK(L, kmax_for(nq), nq, dev)
    for l in range(L):
        s = qn[l] @ qn[l].T
        s.masked_fill_(eye, float("-inf"))
        q_only.merge(l, s, qids)
        for n in sizes:
            state[n].merge(l, s, qids)
    qgal_t = {n: torch.tensor(q_gal[n], device=dev) for n in sizes}

    for si, sf in enumerate(shard_files):
        lo, hi = starts[si], starts[si + 1]
        r = rank[lo:hi]
        feats = torch.load(sf, map_location="cpu", mmap=True)["feats"]
        g = feats[:, layers].to(dev)  # [m, L, D], stored dtype
        del feats
        gids = torch.arange(lo, hi, dtype=torch.int32, device=dev)
        for sidx, n in enumerate(sizes):
            sel = np.nonzero(r <= sidx)[0]
            if len(sel) == 0:
                continue
            sel_t = torch.from_numpy(sel).to(dev)
            gs = g.index_select(0, sel_t) if len(sel) < g.shape[0] else g
            for l in range(L):
                qv = qgal_t[n][l]
                x = gs[:, l].float().clamp_(-qv, qv)
                x = F.normalize(x, dim=-1)
                state[n].merge(l, qn[l] @ x.T, gids[sel_t])
                del x
        del g
        if si % 10 == 9 or si == len(shard_files) - 1:
            print(f"  shard {si + 1}/{len(shard_files)}  {time.time() - t0:.0f}s", flush=True)

    out = {"sizes": [nq] + sizes, "knn": {}, "sim": {}, "layers": list(layer_ids),
           "q_query": q_query, "q_gallery": q_gal, "query_path": args.query_path,
           "shard_dir": args.shard_dir, "query_id0": QUERY_ID0}
    out["knn"][nq], out["sim"][nq] = q_only.finish()
    for n in sizes:
        out["knn"][n], out["sim"][n] = state[n].finish()
        out["sim"][n] = out["sim"][n][:, :, :10].clone()  # keep only what the NN-quality stats use
    out["sim"][nq] = out["sim"][nq][:, :, :10].clone()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(out, args.output)
    print(f"saved {args.output}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
