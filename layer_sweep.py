"""
Mutual kNN for every layer pair of two models at every nested gallery size,
from the kNN sets written by `knn_sets.py`, plus a search-aware null.

Paper App. A.2 (per-scale layer re-selection) and App. C (calibration,
following Groger et al. 2026): captions are re-paired with images at random,
the whole layer grid is rescored, and the grid maximum is taken per
permutation. With tau the upper 5% quantile of that maximum over K
permutations, a score s is reported as s_cal = (s - tau) / (1 - tau).

The shuffle is a block permutation: queries are re-paired among the queries
and gallery rows among that gallery's rows. Model B's neighbour ids are mapped
through the inverse pairing. This is exact for mutual kNN evaluated on the
queries: a query re-paired with another query's caption needs only that
query's stored neighbour set.

Outputs `<output>.npz` (all grids and null grids) and `<output>.json` with,
per size and k:
  fixed:    the WIT-1024 k=10 best pair (the main-text protocol), raw score
  per_k:    pair re-selected at this size for this k   (Fig. S3a), raw / tau / cal
  at_k10:   pair re-selected at this size at k=10, scored at this k (Fig. S3b)

    python layer_sweep.py --a results/knn/<vision>.pt --b results/knn/<language>.pt \
        --output results/alignment/layer_sweep/<vision>__<language>
"""
import argparse
import json
import os
import time

import numpy as np
import torch


def ks_for(n):
    return sorted({1, 10, max(1, n // 100)})


def gallery_members(n, nq, indices_dir):
    if n == nq:
        return np.zeros(0, dtype=np.int64)
    return np.load(os.path.join(indices_dir, f"gallery_indices_{n}.npy")).astype(np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="knn_sets output of model A (e.g. vision)")
    p.add_argument("--b", required=True, help="knn_sets output of model B (e.g. language)")
    p.add_argument("--output", required=True, help="path stem; writes .npz and .json")
    p.add_argument("--perms", type=int, default=200)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gallery_indices_dir", default="data/wit1m/nested_wit1m_indices")
    p.add_argument("--sizes", nargs="*", type=int, default=None)
    args = p.parse_args()

    dev = "cuda"
    t0 = time.time()
    A = torch.load(args.a, map_location="cpu")
    B = torch.load(args.b, map_location="cpu")
    assert A["query_id0"] == B["query_id0"]
    qid0 = A["query_id0"]
    sizes = args.sizes or [n for n in A["sizes"] if n in B["sizes"]]
    nq = A["knn"][A["sizes"][0]].shape[1]
    N = qid0 + nq
    la_ids, lb_ids = A["layers"], B["layers"]
    print(f"A {args.a}: {len(la_ids)} layers   B {args.b}: {len(lb_ids)} layers   sizes {sizes}")

    npz, summary = {}, {"a": args.a, "b": args.b, "perms": args.perms, "alpha": args.alpha,
                        "layers_a": la_ids, "layers_b": lb_ids, "sizes": {}}
    fixed = None
    for n in sorted(sizes):
        ks = ks_for(n)
        kmax = max(ks)
        KA = A["knn"][n][:, :, :kmax].to(dev).long()  # [La, nq, kmax]
        KB = B["knn"][n][:, :, :kmax].to(dev).long()
        La, Lb = KA.shape[0], KB.shape[0]

        # permutation tables: inv[p] maps B's neighbour ids to the sample now holding them
        g = torch.Generator().manual_seed(args.seed + n)
        members = torch.from_numpy(gallery_members(n, nq, args.gallery_indices_dir))
        assert len(members) in (0, n)
        invs, piqs = [], []
        for _ in range(args.perms):
            inv = torch.arange(N, dtype=torch.int32)
            if len(members):
                pg = torch.randperm(len(members), generator=g)
                inv[members[pg]] = members.int()
            pq = torch.randperm(nq, generator=g)
            inv[qid0 + pq] = (qid0 + torch.arange(nq)).int()
            invs.append(inv.to(dev))
            piqs.append(pq.to(dev))

        grid = torch.zeros(La, Lb, len(ks))
        null = torch.zeros(args.perms, La, Lb, len(ks))
        ks_t = torch.tensor(ks, device=dev)
        for a in range(La):
            rank = torch.full((nq, N), 32767, dtype=torch.int16, device=dev)
            rank.scatter_(1, KA[a], torch.arange(kmax, dtype=torch.int16, device=dev).expand(nq, -1).contiguous())

            def score(kb):
                r = rank.gather(1, kb)  # [nq, kmax]: rank in A of each B neighbour
                return torch.stack([(r[:, :k] < k).sum(1).float().mean() / k for k in ks]).cpu()

            for b in range(Lb):
                grid[a, b] = score(KB[b])
                for pi in range(args.perms):
                    kb = invs[pi][KB[b][piqs[pi]]].long()
                    null[pi, a, b] = score(kb)
            del rank
        npz[f"grid_{n}"] = grid.numpy()
        npz[f"null_{n}"] = null.numpy()
        npz[f"ks_{n}"] = np.array(ks)

        # ---- summaries
        qtile = 1 - args.alpha
        flat = grid.reshape(La * Lb, len(ks))
        nflat = null.reshape(args.perms, La * Lb, len(ks))
        i10 = ks.index(10)
        if fixed is None:  # first (smallest) size = WIT-1024: the main-text layer choice
            fixed = int(flat[:, i10].argmax())
        best10 = int(flat[:, i10].argmax())
        null_arg10 = nflat[:, :, i10].argmax(1)  # per permutation, its k=10 pick
        rec = {"ks": ks, "fixed_pair": [la_ids[fixed // Lb], lb_ids[fixed % Lb]],
               "at_k10_pair": [la_ids[best10 // Lb], lb_ids[best10 % Lb]], "k": {}}
        for j, k in enumerate(ks):
            best = int(flat[:, j].argmax())
            s = float(flat[best, j])
            tau = float(torch.quantile(nflat[:, :, j].max(1).values, qtile))
            s10 = float(flat[best10, j])
            tau10 = float(torch.quantile(nflat[torch.arange(args.perms), null_arg10, j], qtile))
            rec["k"][str(k)] = {
                "fixed": float(flat[fixed, j]),
                "per_k": {"pair": [la_ids[best // Lb], lb_ids[best % Lb]], "raw": s, "tau": tau,
                          "cal": (s - tau) / (1 - tau)},
                "at_k10": {"raw": s10, "tau": tau10, "cal": (s10 - tau10) / (1 - tau10)},
                "null_mean": float(nflat[:, :, j].mean()),
            }
        summary["sizes"][str(n)] = rec
        line = "  ".join(f"k={k}: fixed {rec['k'][str(k)]['fixed']:.4f} per-k {rec['k'][str(k)]['per_k']['raw']:.4f}"
                         f"{tuple(rec['k'][str(k)]['per_k']['pair'])} cal {rec['k'][str(k)]['per_k']['cal']:.4f}"
                         for k in ks)
        print(f"n={n:>8,}  {line}  ({time.time() - t0:.0f}s)", flush=True)
        del KA, KB, invs

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez_compressed(args.output + ".npz", layers_a=np.array(la_ids), layers_b=np.array(lb_ids), **npz)
    with open(args.output + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {args.output}.npz/.json")


if __name__ == "__main__":
    main()
