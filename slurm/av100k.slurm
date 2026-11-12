#!/bin/bash -l
#SBATCH --job-name=av100k
#SBATCH --account=jiminxiao
#SBATCH --qos=4gpus
#SBATCH --output=logs/%x-%A_%a.out
#
# Build LAION-Audio-100k / PVD-100k (build_av100k.py). CPU only; one stage per submit, resources on
# the sbatch line (size with `hpcmon fit --gpus 0` first). Run from the checkout root:
#
#   DS=laion_audio STAGE=index    sbatch -J av-index -p <parts> --cpus-per-task=16 --mem=32G --time=1:00:00 --array=0-7 slurm/av100k.slurm
#   DS=laion_audio STAGE=pool     sbatch -J av-pool  -p <parts> --cpus-per-task=2  --mem=32G --time=0:30:00 slurm/av100k.slurm
#   DS=laion_audio STAGE=pack     sbatch -J av-pack  -p <parts> --cpus-per-task=16 --mem=48G --time=2:00:00 --array=0-7 slurm/av100k.slurm
#   DS=laion_audio STAGE=finalize sbatch -J av-final -p <parts> --cpus-per-task=2  --mem=64G --time=0:30:00 slurm/av100k.slurm
#   EXTRA_ARGS="--dedup norm": near-duplicate captions removed too (-> jordangong--laion-audio-100k-nd; reuses the index)
#   DS=pvd: the same four stages (pack = full PyAV decode of each candidate, to count frames).
#
# Array stages split their items round-robin over the array and skip finished outputs.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/repos/plato/cave_umwelten}"
mkdir -p logs
DS=${DS:?set DS=laion_audio|pvd}
STAGE=${STAGE:?set STAGE=index|pool|pack|finalize}
export PLATO_DATA_ROOT=~/datasets
export OMP_NUM_THREADS=1
echo "host=$(hostname) job=${SLURM_JOB_ID:-} task=${SLURM_ARRAY_TASK_ID:-} ds=$DS stage=$STAGE commit=$(git rev-parse --short HEAD)"
~/.local/bin/uv run python build_av100k.py "$DS" "$STAGE" ${EXTRA_ARGS:-}
