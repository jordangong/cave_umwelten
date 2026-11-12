#!/bin/bash -l
#SBATCH --job-name=laion15m
#SBATCH --account=jiminxiao
#SBATCH --qos=4gpus
#SBATCH --output=logs/%x-%A_%a.out
#
# Build LAION-15M (build_laion15m.py). CPU only; one stage per submit, resources on the
# sbatch line (size with `hpcmon fit --gpus 0` first). Run from ~/repos/plato/cave_umwelten:
#
#   STAGE=sample   sbatch -J l15-sample  -p gpu4090,gpu40901t --cpus-per-task=2  --mem=16G --time=0:30:00 slurm/laion15m.slurm
#   STAGE=hash     sbatch -J l15-hash    -p <parts> --cpus-per-task=24 --mem=48G --time=2:00:00 --array=0-15 slurm/laion15m.slurm
#   STAGE=dedup    sbatch -J l15-dedup   -p <parts> --cpus-per-task=4  --mem=96G --time=4:00:00 slurm/laion15m.slurm
#   STAGE=scatter  sbatch -J l15-scatter -p <parts> --cpus-per-task=2  --mem=16G --time=2:00:00 --array=0-7  slurm/laion15m.slurm
#   STAGE=pack     sbatch -J l15-pack    -p <parts> --cpus-per-task=2  --mem=96G --time=2:00:00 --array=0-14 slurm/laion15m.slurm
#   STAGE=finalize sbatch -J l15-final   -p <parts> --cpus-per-task=1  --mem=8G  --time=0:30:00 slurm/laion15m.slurm
#
# The QoS allows 16 submitted jobs per user (array elements count). Array stages split their items round-robin over the array (SLURM_ARRAY_TASK_COUNT), and
# skip finished outputs, so resubmitting a timed-out array is safe.
set -euo pipefail
cd ~/repos/plato/cave_umwelten
mkdir -p logs
STAGE=${STAGE:?set STAGE=sample|hash|dedup|scatter|pack|finalize}
export PLATO_DATA_ROOT=~/datasets
export OMP_NUM_THREADS=1
~/.local/bin/uv run python build_laion15m.py "$STAGE" ${EXTRA_ARGS:-}
