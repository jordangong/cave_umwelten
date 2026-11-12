#!/bin/bash -l
#SBATCH --job-name=cave
#SBATCH --account=jiminxiao
#SBATCH --qos=4gpus
#SBATCH --output=logs/%x-%A_%a.out
#
# Nested WIT-1M / LAION-15M alignment (Koepke et al., "Back into Plato's Cave"), DINOv2-B x OpenLlama-3B.
# One stage per submit; resources go on the sbatch line (size each with `hpcmon fit` first).
# Run from the repo root on hpc-2 (~/repos/plato/cave_umwelten):
#
#   STAGE=vision   sbatch -J cave-vis  -p gpu4090,gpu40901t --gres=gpu:1 --cpus-per-gpu=8 --mem=48G --time=1:00:00 --array=0-100 slurm/cave.slurm
#   STAGE=language sbatch -J cave-lang -p gpua800,gpua8001t,gpu4090,gpu40901t --gres=gpu:1 --cpus-per-gpu=4 --mem=64G --time=1:00:00 --array=0-100 slurm/cave.slurm
#   STAGE=layers   sbatch -J cave-layers -p gpu4090,gpu40901t --cpus-per-task=4 --mem=32G --time=0:20:00 --dependency=afterok:<vis>_0:<lang>_0 slurm/cave.slurm
#   STAGE=align    sbatch -J cave-align  -p gpu4090,gpu40901t --cpus-per-task=16 --mem=96G --time=4:00:00 --array=0-4 --dependency=afterok:<vis>:<lang>:<layers> slurm/cave.slurm
#   STAGE=knn MOD=vision|language sbatch -J cave-knn -p gpu4090,gpu40901t --gres=gpu:1 --cpus-per-task=4 --mem=64G --time=2:00:00 slurm/cave.slurm
#   STAGE=sweep [A=<knn name>] [B=<knn name>] sbatch -J cave-sweep -p gpu4090,gpu40901t --gres=gpu:1 --cpus-per-task=4 --mem=48G --time=2:00:00 slurm/cave.slurm
#   STAGE=plot     sbatch -J cave-plot   -p gpu4090,gpu40901t --cpus-per-task=1 --mem=4G --time=0:10:00 --dependency=afterok:<align> slurm/cave.slurm
#
# vision/language array index a covers items [a*PER_TASK, (a+1)*PER_TASK) of
#   item 0      = wit_1024 (the 1024 queries)
#   item 1..100 = wit_1m shard 0..99 (10k rows each = one parquet file)
# Finished .pt files are skipped, so resubmitting a timed-out array is safe.
# knn: all-layer nested kNN sets of one model -> results/knn/<model>.pt (knn_sets.py, any model/modality).
# sweep: every layer pair of two knn files + search-aware shuffle null (layer_sweep.py; App. A.2, A.1, A.4).
#   A/B default to VIS_NAME / LANG_NAME (LANG_NAME carries _gemini when CAPTION=gemini).
# align array index = gallery size (10K 50K 100K 500K 1M); it uses no GPU (Faiss CPU IndexFlatL2).
#
# DATASET=laion: LAION-15M gallery (ours, docs/laion.md), queries still WIT-1024 (from $Q_DIR).
#   vision/language item i = LAION shard i (0..1499, one parquet file each), no WIT-1024 item;
#   features in results/features/laion_15m/shards/<model>. Only the chosen layers fit on disk:
#   DATASET=laion STAGE=vision   LAYERS=10 PER_TASK=94 sbatch ... --array=0-15 slurm/cave.slurm
#   DATASET=laion STAGE=language LAYERS=23 PADDING=batch PER_TASK=94 sbatch ... --array=0-15 slurm/cave.slurm
#   DATASET=laion STAGE=knn MOD=vision|language  -> results/knn/laion_15m/<model>.pt (sizes 10K..15M)
#   DATASET=laion STAGE=sweep [PERMS=0]          -> results/alignment/layer_sweep_laion15m/A__B.json
#   DATASET=laion STAGE=align --array=0-4        Faiss cross-check up to 1M (5M+ needs n x 16 KB of RAM)
# Q_DIR / Q_LANG override the query features (e.g. batch-padded WIT-1024 language features);
# KNN_NAME overrides the knn output name. Q_OUT redirects WIT-1024 extraction (item 0).
set -euo pipefail
cd ~/repos/plato/cave_umwelten
mkdir -p logs

STAGE=${STAGE:?set STAGE=vision|language|layers|align|knn|sweep|plot}
DATASET=${DATASET:-wit}                  # wit (WIT-1M gallery) | laion (LAION-15M gallery)
PER_TASK=${PER_TASK:-1}
VIS_MODEL=${VIS_MODEL:-vit_base_patch14_dinov2.lvd142m}
LANG_MODEL=${LANG_MODEL:-openlm-research/open_llama_3b}
VIS_BS=${VIS_BS:-64}
LANG_BS=${LANG_BS:-16}
CAPTION=${CAPTION:-original}             # original | gemini (language features only)
PADDING=${PADDING:-shard}                # shard (upstream) | batch (length-sorted, ~25x cheaper on original captions)
TOKEN_BUDGET=${TOKEN_BUDGET:-8192}       # --padding batch: padded tokens per batch
LAYERS=${LAYERS:-}                       # extract only these layer positions (space-separated); default all
CAP_TAG=$([ "$CAPTION" = gemini ] && echo _gemini || true)

# everything is pre-fetched: checkpoints in ~/checkpoints, parquet in ~/datasets/{askoepke--wit_1m_recaptioned,jordangong--laion-15m}
export PLATO_CKPT_ROOT=~/checkpoints PLATO_DATA_ROOT=~/datasets

FEAT=results/features
Q_DIR=${Q_DIR:-$FEAT/wit_1024}
Q_OUT=${Q_OUT:-$FEAT/wit_1024}
Q_VIS=${Q_VIS:-$Q_DIR/${VIS_MODEL}_pool-cls.pt}
Q_LANG=${Q_LANG:-$Q_DIR/${LANG_MODEL//\//_}_pool-avg${CAP_TAG}.pt}
BEST=results/best_layers${CAP_TAG}.json
VIS_NAME=${VIS_MODEL//\//_}
LANG_NAME=${LANG_MODEL//\//_}${CAP_TAG}
case $DATASET in
  wit)
    GAL=wit_1m; NUM_SHARDS=100; FIRST_SHARD_ITEM=1     # item 0 = wit_1024, item i = shard i-1
    IDX_DIR=./data/wit1m/nested_wit1m_indices
    SIZES=(10000 50000 100000 500000 1000000)
    ALIGN_DIR=results/alignment/nested_wit1m; ALIGN_NAME=nested_wit1m
    KNN_DIR=results/knn; SWEEP_DIR=results/alignment/layer_sweep; PERMS=${PERMS:-200} ;;
  laion)
    [ "$CAPTION" = original ] || { echo "laion_15m has original captions only"; exit 1; }
    GAL=laion_15m; NUM_SHARDS=1500; FIRST_SHARD_ITEM=0  # item i = shard i
    IDX_DIR=$PLATO_DATA_ROOT/jordangong--laion-15m/nested_indices
    SIZES=(10000 50000 100000 500000 1000000 5000000 10000000 15000000)
    ALIGN_DIR=results/alignment/nested_laion15m; ALIGN_NAME=nested_laion15m
    KNN_DIR=results/knn/laion_15m; SWEEP_DIR=results/alignment/layer_sweep_laion15m; PERMS=${PERMS:-0} ;;
  *) echo "unknown DATASET $DATASET"; exit 1 ;;
esac
NUM_ITEMS=$((NUM_SHARDS + FIRST_SHARD_ITEM))
G_VIS=$FEAT/$GAL/shards/${VIS_MODEL}
G_LANG=$FEAT/$GAL/shards/${LANG_MODEL//\//_}${CAP_TAG}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-$(nproc)}
UV=~/.local/bin/uv
TASK=${SLURM_ARRAY_TASK_ID:-0}

echo "host=$(hostname) job=${SLURM_JOB_ID:-} task=$TASK stage=$STAGE dataset=$DATASET caption=$CAPTION commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

extract() {  # extract <modality> <item>
  local mod=$1 item=$2 extra=()
  if [ "$mod" = vision ]; then
    extra=(--model "$VIS_MODEL" --batch_size "$VIS_BS" --num_workers "${SLURM_CPUS_PER_TASK:-8}")
  else
    extra=(--model "$LANG_MODEL" --batch_size "$LANG_BS" --caption "$CAPTION"
           --padding "$PADDING" --token_budget "$TOKEN_BUDGET")
  fi
  [ -n "$LAYERS" ] && extra+=(--layer_indices $LAYERS)
  if [ "$item" -lt "$FIRST_SHARD_ITEM" ]; then
    $UV run python extract_features.py --modality "$mod" --config wit_1024 \
      --output_dir "$Q_OUT" "${extra[@]}"
  else
    $UV run python extract_features.py --modality "$mod" --config $GAL \
      --shard_id $((item - FIRST_SHARD_ITEM)) --num_shards $NUM_SHARDS \
      --output_dir $FEAT/$GAL/shards "${extra[@]}"
  fi
}

case $STAGE in
  vision|language)
    for ((item = TASK * PER_TASK; item < (TASK + 1) * PER_TASK && item < NUM_ITEMS; item++)); do
      extract "$STAGE" "$item"
    done ;;
  layers)
    $UV run python find_best_layers.py --vision_path "$Q_VIS" --language_path "$Q_LANG" --output "$BEST" ;;
  align)
    read -r DL LL < <($UV run python -c "import json; b=json.load(open('$BEST')); print(b['vision_layer'], b['language_layer'])")
    echo "best layers: vision $DL, language $LL (from $BEST)"
    $UV run python align_nested_wit1m.py \
      --dino_layer "$DL" --llm_layer "$LL" --caption_type "$CAPTION" \
      --query_dino_path "$Q_VIS" --query_llm_path "$Q_LANG" \
      --gallery_dino_shard_dir "$G_VIS" --gallery_llm_shard_dir "$G_LANG" \
      --gallery_indices_dir "$IDX_DIR" --name $ALIGN_NAME \
      --output_dir $ALIGN_DIR --sizes "${SIZES[$TASK]}" ;;
  knn)
    if [ "${MOD:?set MOD=vision|language}" = vision ]; then QP=$Q_VIS; SD=$G_VIS; NAME=$VIS_NAME; else QP=$Q_LANG; SD=$G_LANG; NAME=$LANG_NAME; fi
    NAME=${KNN_NAME:-$NAME}
    $UV run python knn_sets.py --query_path "$QP" --shard_dir "$SD" \
      --gallery_indices_dir "$IDX_DIR" --sizes "${SIZES[@]}" --output $KNN_DIR/$NAME.pt ;;
  sweep)
    A=${A:-$VIS_NAME}; B=${B:-$LANG_NAME}
    $UV run python layer_sweep.py --a $KNN_DIR/$A.pt --b $KNN_DIR/$B.pt \
      --gallery_indices_dir "$IDX_DIR" \
      --perms "$PERMS" --output $SWEEP_DIR/${A}__${B} ;;
  plot)
    $UV run python plot_nested_evolution.py --result_dir $ALIGN_DIR \
      --pattern "${ALIGN_NAME}${CAP_TAG}_size*.npy" --output $ALIGN_DIR/plot_nested_evolution${CAP_TAG}.png ;;
  *) echo "unknown STAGE $STAGE"; exit 1 ;;
esac
echo "done $(date)"
