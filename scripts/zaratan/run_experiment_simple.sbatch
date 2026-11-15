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
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=standard

set -euo pipefail

CONDA_HOME="${CONDA_HOME:-$HOME/miniforge3}"
ENV_NAME="${INFRARL_ENV:-infrarl}"

# shellcheck disable=SC1091
source "$CONDA_HOME/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# single-node BLAS/torch threading
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TOKENIZERS_PARALLELISM=false

mkdir -p logs results
echo "[run] host=$(hostname) env=$CONDA_DEFAULT_ENV cpus=${SLURM_CPUS_PER_TASK:-1}"
echo "[run] git sha=$(git rev-parse --short HEAD 2>/dev/null || echo none)"
# 8 agents: random, heuristic, context-free + LinUCB + D-LinUCB bandits,
# Q-learning, PPO, and Lagrangian PPO (CMDP constrained variant).
# Ablation sweep on lagrangian_ppo (5 variants x 3 seeds = 15 training runs).
# Out dir INFRARL_OUT can be overridden via --export at sbatch time.
: "${INFRARL_OUT:=results_frozen_v1}"
python -m infrarl.evaluation.run_experiment \
  --agents random,heuristic,bandit_context_free,bandit_contextual,bandit_discounted,q_learning,ppo,lagrangian_ppo \
  --seeds 3 \
  --ablate lagrangian_ppo \
  --out "$INFRARL_OUT"
echo "[run] done at $(date)"
