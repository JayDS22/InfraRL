#!/bin/bash
# SLURM job: full 3-seed experiment with PPO ablation sweep.
#
# Submit from the repo root:
#   sbatch scripts/zaratan/run_experiment.sbatch
#
# Tune the resource directives below based on your Zaratan allocation. Defaults
# assume standard partition, single CPU task, ~90 min wall time (the laptop
# baseline). Bump --cpus-per-task to use more BLAS threads inside numpy/torch.

#SBATCH --job-name=infrarl-full
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=standard

set -euo pipefail

module purge
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${INFRARL_ENV:-infrarl}"

# single-node BLAS/torch threading
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TOKENIZERS_PARALLELISM=false

mkdir -p logs results
echo "[run] host=$(hostname) env=$CONDA_DEFAULT_ENV cpus=$SLURM_CPUS_PER_TASK"
echo "[run] git sha=$(git rev-parse --short HEAD 2>/dev/null || echo none)"
python -m infrarl.evaluation.run_experiment --seeds 3 --ablate ppo --out results
echo "[run] done at $(date)"
