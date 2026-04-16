#!/bin/bash
#SBATCH --job-name=extract_features
#SBATCH --output=logs/extract_features_%A_%a.out
#SBATCH --error=logs/extract_features_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-101

# ============================================================
# Extract DINO + OpenLLaMA features on WIT-1024 and WIT-1M.
#
# Array task layout (102 tasks):
#   0        : wit_1024 vision (full, 1024 samples)
#   1        : wit_1024 language (full, 1024 samples)
#   2..51    : wit_1m   vision   shards 0..49  (num_shards=50, 20k each)
#   52..101  : wit_1m   language shards 0..49  (num_shards=50, 20k each)
#
# Adapt the environment activation below to your setup.
# Submit from the repo root: sbatch slurm_scripts/slurm_extract_features_generic.sh
# ============================================================

NUM_SHARDS=50

echo "Job started at: $(date)"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"

# cd to repo root (parent of slurm_scripts/)
cd "$(dirname "$0")/.." || exit 1

mkdir -p logs

# --- Activate environment (edit to match your setup) ---
# conda activate modality_caves
# --------------------------------------------------------

TASK=$SLURM_ARRAY_TASK_ID

if [ "$TASK" -eq 0 ]; then
    python extract_features.py --modality vision --config wit_1024 \
        --output_dir results/features/wit_1024
elif [ "$TASK" -eq 1 ]; then
    python extract_features.py --modality language --config wit_1024 \
        --output_dir results/features/wit_1024
elif [ "$TASK" -ge 2 ] && [ "$TASK" -le 51 ]; then
    SHARD=$((TASK - 2))
    python extract_features.py --modality vision --config wit_1m \
        --shard_id "$SHARD" --num_shards "$NUM_SHARDS" \
        --output_dir results/features/wit_1m/shards
elif [ "$TASK" -ge 52 ] && [ "$TASK" -le 101 ]; then
    SHARD=$((TASK - 52))
    python extract_features.py --modality language --config wit_1m \
        --shard_id "$SHARD" --num_shards "$NUM_SHARDS" \
        --output_dir results/features/wit_1m/shards
else
    echo "ERROR: unknown array task id $TASK"
    exit 1
fi

echo "Job completed at: $(date)"
