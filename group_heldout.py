"""
Mutual kNN with each query's own group held out, from knn_sets.py outputs.

On LAION-Audio-100k the fine-grained neighbours of a query are mostly clips from
the same Freesound uploader (sample-pack siblings such as "kama swipe 4." /
"kama swipe 5."), and both modalities find them, so k=1 mutual kNN rises with
gallery size. Here every neighbour from the query's group is skipped in both
lists before the top-k is compared; queries without k such neighbours in a
stored list are left out (the stored lists are max(10, n/100) deep).

Groups (--group uploader): Freesound username from `source_key`; every clip of
another source is its own group.

    python group_heldout.py --dataset laion_audio_100k \
        --a results/knn/laion_audio_100k/mispeech_dasheng-base.pt \
        --b results/knn/laion_audio_100k/google_gemma-2-9b.pt \
        --sweep results/alignment/layer_sweep_laion_audio_100k/mispeech_dasheng-base__google_gemma-2-9b.json \
        --output results/alignment/heldout_laion_audio_100k.json
"""
import argparse
import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq
import torch

import paths

N_QUERY = 1000


def groups(dataset):
    files = sorted(glob.glob(os.path.join(paths.data_root(), "jordangong--" + dataset.replace("_", "-"),
                                          dataset, "*.parquet")))
    meta = pq.read_table(files, columns=["source", "source_key"]).to_pylist()
    return np.array([m["source_key"].split("|")[-1] if m["source"] == "freesound" else f"{m['source']}:{i}"
                     for i, m in enumerate(meta)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="laion_audio_100k or laion_audio_100k_nd")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--sweep", required=True, help="layer_sweep json; its fixed pair is used")
    p.add_argument("--ks", nargs="*", type=int, default=[1, 10])
    p.add_argument("--output", required=True)
    args = p.parse_args()

    A = torch.load(args.a, map_location="cpu", weights_only=False)
    B = torch.load(args.b, map_location="cpu", weights_only=False)
    assert A["size_includes_queries"] and A["query_id0"] == B["query_id0"]
    sw = json.load(open(args.sweep))
    fa, fb = next(iter(sw["sizes"].values()))["fixed_pair"]
    la, lb = A["layers"].index(fa), B["layers"].index(fb)
    qid0 = A["query_id0"]
    grp = groups(args.dataset)

    def group_of(ids):
        ids = ids.numpy()
        return grp[np.where(ids >= qid0, ids - qid0, N_QUERY + ids)]

    out = {"dataset": args.dataset, "a": args.a, "b": args.b, "fixed_pair": [fa, fb], "sizes": {}}
    for n in A["sizes"]:
        KA, KB = A["knn"][n][la].long(), B["knn"][n][lb].long()
        rec = {"raw_k1": float((KA[:, 0] == KB[:, 0]).float().mean())}
        same_a = same_b = 0
        for j in range(N_QUERY):
            same_a += int(group_of(KA[j, :1])[0] == grp[j])
            same_b += int(group_of(KB[j, :1])[0] == grp[j])
        rec["top1_same_group"] = {"a": same_a / N_QUERY, "b": same_b / N_QUERY}
        for k in args.ks:
            hits, valid = 0.0, 0
            for j in range(N_QUERY):
                na = KA[j][torch.from_numpy(group_of(KA[j]) != grp[j])][:k]
                nb = KB[j][torch.from_numpy(group_of(KB[j]) != grp[j])][:k]
                if len(na) < k or len(nb) < k:
                    continue
                valid += 1
                hits += len(set(na.tolist()) & set(nb.tolist())) / k
            rec[f"heldout_k{k}"] = {"score": hits / max(valid, 1), "queries": valid}
        out["sizes"][str(n)] = rec
        print(f"n={n:>7}  raw k=1 {rec['raw_k1']:.4f}  top-1 same group a {same_a / N_QUERY:.3f} "
              f"b {same_b / N_QUERY:.3f}  " + "  ".join(
                  f"held-out k={k} {rec[f'heldout_k{k}']['score']:.4f} ({rec[f'heldout_k{k}']['queries']} q)"
                  for k in args.ks))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
