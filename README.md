<h1 align="center">Back into Plato's Cave: Examining Cross-modal Representational Convergence at Scale</h1>

<p align="center">
<a href="https://akoepke.github.io/">A. Sophia Koepke</a>, <a href="https://scholar.google.com/citations?user=-4l5Il4AAAAJ">Daniil Zverev</a>, <a href="http://shiry.ttic.edu/">Shiry Ginosar</a>, <a href="https://www2.eecs.berkeley.edu/Faculty/Homepages/efros.html">Alexei A. Efros</a>
</p>

<br>

<p align="center">
<a href="https://akoepke.github.io/cave_umwelten">Project Page</a> | <a href="https://akoepke.github.io/cave_umwelten/koepke2026_cave_umwelten.pdf">Paper</a>
</p>


## Requirements

```bash
conda create -n cave_umwelten python=3.10 -y
conda activate cave_umwelten

# PyTorch (CUDA 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# FAISS (GPU build; use CPU build if no CUDA is available)
conda install -c conda-forge faiss-gpu -y

pip install -r requirements.txt
```

## Data

Features are extracted directly from the HuggingFace dataset
`askoepke/wit_1m_recaptioned`, which has two configs:

- `wit_1024` — 1024 held-out query samples
- `wit_1m`   — 1M gallery samples

No manual download is needed; `extract_features.py` streams the parquet shards
via `datasets.load_dataset`.

## Feature extraction

Features are saved at every transformer layer so any layer pair can be
selected at alignment time.

```bash
# Queries (WIT-1024, full)
python extract_features.py --modality vision   --config wit_1024 \
    --output_dir results/features/wit_1024
python extract_features.py --modality language --config wit_1024 \
    --output_dir results/features/wit_1024

# Gallery (WIT-1M, sharded — run one shard at a time or as a SLURM array)
python extract_features.py --modality vision   --config wit_1m \
    --shard_id 0 --num_shards 50 \
    --output_dir results/features/wit_1m/shards
python extract_features.py --modality language --config wit_1m \
    --shard_id 0 --num_shards 50 \
    --output_dir results/features/wit_1m/shards
```

Defaults: `vit_base_patch14_dinov2.lvd142m` for vision,
`openlm-research/open_llama_3b` for language, `original` captions. Override
with `--model` and `--caption`. See `slurm_scripts/slurm_extract_features_generic.sh` for the
102-task array that produces the full WIT-1024 + sharded WIT-1M set.

## Picking layers

`find_best_layers.py` scans all (vision, language) layer pairs on WIT-1024
and writes the best-aligning pair to `best_layers.json`:

```bash
python find_best_layers.py \
    --vision_path   results/features/wit_1024/vit_base_patch14_dinov2.lvd142m_pool-cls.pt \
    --language_path results/features/wit_1024/openlm-research_open_llama_3b_pool-avg.pt
```

## Nested alignment

`align_nested_wit1m.py` computes mutual-kNN alignment on nested WIT-1M
gallery subsets (10K → 1M). Precomputed `gallery_indices_*.npy` in
`data/wit1m/nested_wit1m_indices/` define the nesting; each smaller set is a strict subset of the next.

```bash
python align_nested_wit1m.py \
    --dino_layer 10 --llm_layer 23 \
    --query_dino_path ./results/features/wit_1024/vit_base_patch14_dinov2.lvd142m_pool-cls.pt \
    --query_llm_path  ./results/features/wit_1024/openlm-research_open_llama_3b_pool-avg.pt \
    --gallery_dino_shard_dir ./results/features/wit_1m/shards/vit_base_patch14_dinov2.lvd142m \
    --gallery_llm_shard_dir  ./results/features/wit_1m/shards/openlm-research_open_llama_3b \
    --gallery_indices_dir ./data/wit1m/nested_wit1m_indices \
    --output_dir ./results/alignment/nested_wit1m
```

Use `--sizes 10000` to run a single size (useful for smoke tests or one
SLURM task per size — see `slurm_scripts/slurm_align_nested_wit1m_generic.sh`).

## Plotting

```bash
python plot_nested_evolution.py \
    --result_dir results/alignment/nested_wit1m \
    --output     results/alignment/nested_wit1m/plot_nested_evolution.png
```

## Citation

If you find this code useful, please cite:

```bibtex
@article{koepke2026cave,
  title={Back into Plato's Cave: Examining Cross-modal Representational Convergence at Scale},
  author={Koepke, A. Sophia and Zverev, Daniil and Ginosar, Shiry and Efros, Alexei A.},
  year={2026}
}
```


This work builds on the [Platonic Representation Hypothesis](https://github.com/minyoungg/platonic-rep).
