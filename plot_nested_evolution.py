"""
Plot nested-neighborhood-evolution: mutual-kNN alignment vs gallery size,
one curve per k choice (k=1, k=10, k=n/100).

Reads the per-size result .npy files written by align_nested_wit1m.py.
"""
import argparse
import glob
import os

import numpy as np
import matplotlib.pyplot as plt


def load_results(result_dir):
    """Find all per-size result files and aggregate them."""
    files = sorted(glob.glob(os.path.join(result_dir, "nested_wit1m*_size*.npy")))
    if not files:
        raise FileNotFoundError(f"No nested_wit1m*_size*.npy files under {result_dir}")
    rows = []
    for f in files:
        d = np.load(f, allow_pickle=True).item()
        for size in sorted(d["results"].keys()):
            r = d["results"][size]
            rows.append({
                "size": size,
                "k_pct": r["k_pct"],
                "k1": r["score_k1"],
                "k10": r["score_k10"],
                "kp": r["score_k_pct"],
            })
    rows.sort(key=lambda x: x["size"])
    # dedupe: keep last result for each size
    seen = {}
    for r in rows:
        seen[r["size"]] = r
    return sorted(seen.values(), key=lambda r: r["size"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result_dir", default="results/alignment/nested_wit1m")
    p.add_argument("--output", default="results/alignment/nested_wit1m/plot_nested_evolution.png")
    p.add_argument("--title", default="Nested WIT-1M: mutual kNN alignment")
    args = p.parse_args()

    rows = load_results(args.result_dir)
    sizes = [r["size"] for r in rows]

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.8))
    ax.plot(sizes, [r["k1"] for r in rows], "o-", label="k=1")
    ax.plot(sizes, [r["k10"] for r in rows], "s-", label="k=10")
    ax.plot(sizes, [r["kp"] for r in rows], "^-", label="k=n/100")
    ax.set_xscale("log")
    ax.set_xlabel("Gallery size")
    ax.set_ylabel("Mutual kNN score (higher = better)")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved {args.output}")

    print(f"\n{'size':>10} {'k_pct':>6} | {'k=1':>7} {'k=10':>7} {'k=n/100':>9}")
    for r in rows:
        print(f"{r['size']:>10,} {r['k_pct']:>6} | {r['k1']:>7.4f} {r['k10']:>7.4f} {r['kp']:>9.4f}")


if __name__ == "__main__":
    main()
