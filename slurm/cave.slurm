#!/bin/bash -l
#SBATCH --job-name=cave
#SBATCH --account=jiminxiao
#SBATCH --qos=4gpus
#SBATCH --output=logs/%x-%A_%a.out
#
# Nested WIT-1M alignment (Koepke et al., "Back into Plato's Cave"), DINOv2-B x OpenLlama-3B.
# One stage per submit; resources go on the sbatch line (size each with `hpcmon fit` first).
# Run from the repo root on hpc-2 (~/repos/plato/cave_umwelten):
#
#   STAGE=vision   sbatch -J cave-vis  -p gpu4090,gpu40901t --gres=gpu:1 --cpus-per-gpu=8 --mem=48G --time=1:00:00 --array=0-100 slurm/cave.slurm
#   STAGE=language sbatch -J cave-lang -p gpua800,gpua8001t,gpu4090,gpu40901t --gres=gpu:1 --cpus-per-gpu=4 --mem=64G --time=1:00:00 --array=0-100 slurm/cave.slurm
#   STAGE=layers   sbatch -J cave-layers -p gpu4090,gpu40901t --cpus-per-task=4 --mem=32G --time=0:20:00 --dependency=afterok:<vis>_0:<lang>_0 slurm/cave.slurm
#   STAGE=align    sbatch -J cave-align  -p gpu4090,gpu40901t --cpus-per-task=16 --mem=96G --time=4:00:00 --array=0-4 --dependency=afterok:<vis>:<lang>:<layers> slurm/cave.slurm
#   STAGE=plot     sbatch -J cave-plot   -p gpu4090,gpu40901t --cpus-per-task=1 --mem=4G --time=0:10:00 --dependency=afterok:<align> slurm/cave.slurm
#
# vision/language array index a covers items [a*PER_TASK, (a+1)*PER_TASK) of
#   item 0      = wit_1024 (the 1024 queries)
#   item 1..100 = wit_1m shard 0..99 (10k rows each = one parquet file)
# Finished .pt files are skipped, so resubmitting a timed-out array is safe.
# align array index = gallery size (10K 50K 100K 500K 1M); it uses no GPU (Faiss CPU IndexFlatL2).
set -euo pipefail
cd ~/repos/plato/cave_umwelten
mkdir -p logs

STAGE=${STAGE:?set STAGE=vision|language|layers|align|plot}
PER_TASK=${PER_TASK:-1}
NUM_SHARDS=100
VIS_MODEL=vit_base_patch14_dinov2.lvd142m
LANG_MODEL=openlm-research/open_llama_3b
VIS_BS=${VIS_BS:-64}
LANG_BS=${LANG_BS:-16}
CAPTION=${CAPTION:-original}             # original | gemini (language features only)
CAP_TAG=$([ "$CAPTION" = gemini ] && echo _gemini || true)

FEAT=results/features
Q_VIS=$FEAT/wit_1024/${VIS_MODEL}_pool-cls.pt
Q_LANG=$FEAT/wit_1024/${LANG_MODEL//\//_}_pool-avg${CAP_TAG}.pt
G_VIS=$FEAT/wit_1m/shards/${VIS_MODEL}
G_LANG=$FEAT/wit_1m/shards/${LANG_MODEL//\//_}${CAP_TAG}
BEST=results/best_layers${CAP_TAG}.json
ALIGN_DIR=results/alignment/nested_wit1m

# everything is pre-fetched: checkpoints in ~/checkpoints, parquet in ~/datasets/askoepke--wit_1m_recaptioned
export PLATO_CKPT_ROOT=~/checkpoints PLATO_DATA_ROOT=~/datasets
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-$(nproc)}
UV=~/.local/bin/uv
TASK=${SLURM_ARRAY_TASK_ID:-0}

echo "host=$(hostname) job=${SLURM_JOB_ID:-} task=$TASK stage=$STAGE caption=$CAPTION commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

extract() {  # extract <modality> <item>
  local mod=$1 item=$2 extra=()
  if [ "$mod" = vision ]; then
    extra=(--batch_size "$VIS_BS" --num_workers "${SLURM_CPUS_PER_TASK:-8}")
  else
    extra=(--batch_size "$LANG_BS" --caption "$CAPTION")
  fi
  if [ "$item" -eq 0 ]; then
    $UV run python extract_features.py --modality "$mod" --config wit_1024 \
      --output_dir $FEAT/wit_1024 "${extra[@]}"
  else
    $UV run python extract_features.py --modality "$mod" --config wit_1m \
      --shard_id $((item - 1)) --num_shards $NUM_SHARDS \
      --output_dir $FEAT/wit_1m/shards "${extra[@]}"
  fi
}

case $STAGE in
  vision|language)
    for ((item = TASK * PER_TASK; item < (TASK + 1) * PER_TASK && item <= NUM_SHARDS; item++)); do
      extract "$STAGE" "$item"
    done ;;
  layers)
    $UV run python find_best_layers.py --vision_path "$Q_VIS" --language_path "$Q_LANG" --output "$BEST" ;;
  align)
    SIZES=(10000 50000 100000 500000 1000000)
    read -r DL LL < <($UV run python -c "import json; b=json.load(open('$BEST')); print(b['vision_layer'], b['language_layer'])")
    echo "best layers: vision $DL, language $LL (from $BEST)"
    $UV run python align_nested_wit1m.py \
      --dino_layer "$DL" --llm_layer "$LL" --caption_type "$CAPTION" \
      --query_dino_path "$Q_VIS" --query_llm_path "$Q_LANG" \
      --gallery_dino_shard_dir "$G_VIS" --gallery_llm_shard_dir "$G_LANG" \
      --gallery_indices_dir ./data/wit1m/nested_wit1m_indices \
      --output_dir $ALIGN_DIR --sizes "${SIZES[$TASK]}" ;;
  plot)
    $UV run python plot_nested_evolution.py --result_dir $ALIGN_DIR \
      --pattern "nested_wit1m${CAP_TAG}_size*.npy" --output $ALIGN_DIR/plot_nested_evolution${CAP_TAG}.png ;;
  *) echo "unknown STAGE $STAGE"; exit 1 ;;
esac
echo "done $(date)"
