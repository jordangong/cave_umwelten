"""
Plots from layer_sweep.py summaries (results/alignment/layer_sweep/*.json).

  reselect: one pair, Fig. S3 style. (a) layer pair re-selected per size and
            per k, (b) re-selected per size at k=10 and applied to every k.
            Solid = raw, dotted = calibrated (s - tau) / (1 - tau); the thin
            grey line is the fixed WIT-1024 pair (main-text protocol).
  pairs:    several pairs, fixed WIT-1024 pair (Fig. 4 / S1 / S5-S7 style),
            one panel per k in {1, 10, n/100}.
  scaling:  one panel per pair, k=1 / 10 / n/100 of the fixed pair against
            gallery size (Fig. 4 / Fig. 6a-b style; for text-audio/video the
            fixed pair is chosen at the 1k query level).

    python plot_layer_sweep.py reselect results/alignment/layer_sweep/A__B.json -o out.png
    python plot_layer_sweep.py pairs results/alignment/layer_sweep/{A__B,C__D}.json -o out.png
    python plot_layer_sweep.py scaling results/alignment/layer_sweep_{laion_audio,pvd}_100k/*.json -o out.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KEYS = [("k=1", lambda n: 1), ("k=10", lambda n: 10), ("k=n/100", lambda n: max(1, n // 100))]


def short(path):
    return os.path.basename(path).replace(".pt", "").replace("openlm-research_", "").replace(".lvd142m", "")


def plot_reselect(path, out):
    rec = json.load(open(path))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    colors = dict(zip([k for k, _ in KEYS], ["C0", "C1", "C2"]))
    for ax, mode, title in [(axes[0], "per_k", "(a) re-selected per size and per k"),
                            (axes[1], "at_k10", "(b) re-selected per size at k=10")]:
        for name, kf in KEYS:
            xs, raw, cal, fix = [], [], [], []
            for n_str, r in sorted(rec["sizes"].items(), key=lambda t: int(t[0])):
                n = int(n_str)
                k = str(kf(n))
                if k not in r["k"] or (name == "k=n/100" and n // 100 < 10):
                    continue
                xs.append(n)
                raw.append(r["k"][k][mode]["raw"])
                cal.append(r["k"][k][mode]["cal"])
                fix.append(r["k"][k]["fixed"])
            ax.plot(xs, raw, "o-", color=colors[name], label=f"{name}")
            ax.plot(xs, cal, "o:", color=colors[name], alpha=0.7)
            ax.plot(xs, fix, "-", color="grey", lw=0.8, alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("gallery size")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mutual kNN")
    axes[0].legend(title="solid raw, dotted calibrated,\ngrey fixed WIT-1024 pair", fontsize=8, title_fontsize=8)
    fig.suptitle(f"{short(rec['a'])} x {short(rec['b'])}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


def plot_pairs(paths, out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for path in paths:
        rec = json.load(open(path))
        fp = next(iter(rec["sizes"].values()))["fixed_pair"]
        label = f"{short(rec['a'])} L{fp[0]} x {short(rec['b'])} L{fp[1]}"
        for ax, (name, kf) in zip(axes, KEYS):
            xs, ys = [], []
            for n_str, r in sorted(rec["sizes"].items(), key=lambda t: int(t[0])):
                n = int(n_str)
                k = str(kf(n))
                if k in r["k"]:
                    xs.append(n)
                    ys.append(r["k"][k]["fixed"])
            ax.plot(xs, ys, "o-", label=label)
    for ax, (name, _) in zip(axes, KEYS):
        ax.set_xscale("log")
        ax.set_xlabel("gallery size")
        ax.set_title(name)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mutual kNN (layer pair fixed at WIT-1024)")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


def plot_scaling(paths, out):
    fig, axes = plt.subplots(1, len(paths), figsize=(5.5 * len(paths), 4), squeeze=False)
    for ax, path in zip(axes[0], paths):
        rec = json.load(open(path))
        fp = next(iter(rec["sizes"].values()))["fixed_pair"]
        for name, kf in KEYS:
            xs, ys = [], []
            for n_str, r in sorted(rec["sizes"].items(), key=lambda t: int(t[0])):
                n = int(n_str)
                k = str(kf(n))
                if k in r["k"] and not (name == "k=n/100" and n // 100 < 10):
                    xs.append(n)
                    ys.append(r["k"][k]["fixed"])
            ax.plot(xs, ys, "o-", label=name)
        ax.set_xscale("log")
        ax.set_xlabel("gallery size")
        ax.set_title(f"{short(rec['a'])} L{fp[0]} x {short(rec['b'])} L{fp[1]}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("mutual kNN (fixed pair)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["reselect", "pairs", "scaling"])
    p.add_argument("paths", nargs="+")
    p.add_argument("-o", "--output", required=True)
    args = p.parse_args()
    if args.mode == "reselect":
        plot_reselect(args.paths[0], args.output)
    elif args.mode == "pairs":
        plot_pairs(args.paths, args.output)
    else:
        plot_scaling(args.paths, args.output)
