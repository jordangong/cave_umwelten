#!/bin/bash
#SBATCH --job-name=align_nested
#SBATCH --output=logs/align_nested_%A_%a.out
#SBATCH --error=logs/align_nested_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=768G
#SBATCH --array=0-4

# ============================================================
# Nested alignment: WIT-1024 queries vs WIT-1M gallery subsets.
# Gallery sizes: 10K, 50K, 100K, 500K, 1M (one per array task).
# Layer pair is read from best_layers.json (written by
# find_best_layers.py before submission).
#
# Adapt the environment activation below to your setup.
# Submit from the repo root: sbatch slurm_scripts/slurm_align_nested_wit1m_generic.sh
# ============================================================

echo "Job started at: $(date)"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"

# cd to repo root (parent of slurm_scripts/)
cd "$(dirname "$0")/.." || exit 1

mkdir -p logs

# --- Activate environment (edit to match your setup) ---
# conda activate modality_caves
# --------------------------------------------------------

BEST_LAYERS_JSON="best_layers.json"
if [ ! -f "$BEST_LAYERS_JSON" ]; then
    echo "ERROR: $BEST_LAYERS_JSON not found. Run find_best_layers.py first."
    exit 1
fi

DINO_LAYER=$(python -c "import json; print(json.load(open('$BEST_LAYERS_JSON'))['vision_layer'])")
LLM_LAYER=$(python -c "import json; print(json.load(open('$BEST_LAYERS_JSON'))['language_layer'])")
echo "Using DINO layer $DINO_LAYER, LLM layer $LLM_LAYER"

SIZES=(10000 50000 100000 500000 1000000)
SIZE=${SIZES[$SLURM_ARRAY_TASK_ID]}

python align_nested_wit1m.py \
    --dino_layer "$DINO_LAYER" --llm_layer "$LLM_LAYER" \
    --query_dino_path ./results/features/wit_1024/vit_base_patch14_dinov2.lvd142m_pool-cls.pt \
    --query_llm_path  ./results/features/wit_1024/openlm-research_open_llama_3b_pool-avg.pt \
    --gallery_dino_shard_dir ./results/features/wit_1m/shards/vit_base_patch14_dinov2.lvd142m_pool-cls \
    --gallery_llm_shard_dir  ./results/features/wit_1m/shards/openlm-research_open_llama_3b_pool-avg \
    --gallery_indices_dir ./data/wit1m/nested_wit1m_indices \
    --sizes $SIZE

echo "Job completed at: $(date)"
