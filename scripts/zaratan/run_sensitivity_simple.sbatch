#!/bin/bash
# SLURM job: simulator-perturbation sensitivity sweep.
#
# Trains each agent once per seed under the nominal estimator, then evaluates
# on the held-out test profiles under multiplicative perturbations of
# (quality, latency, cost). Produces results_sensitivity/sensitivity.csv and
# sensitivity.json, consumed by plots.plot_sensitivity() for the ranking plot.
#
# Submit from the repo root:
#   sbatch scripts/zaratan/run_sensitivity_simple.sbatch
#
# Override agents or output dir via sbatch --export at submit time, e.g.:
#   sbatch --export=INFRARL_AGENTS=heuristic,ppo,lagrangian_ppo,INFRARL_OUT=results_sens_v1 \
#          scripts/zaratan/run_sensitivity_simple.sbatch

#SBATCH --job-name=infrarl-sens
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TOKENIZERS_PARALLELISM=false

mkdir -p logs results
echo "[sens] host=$(hostname) env=$CONDA_DEFAULT_ENV cpus=${SLURM_CPUS_PER_TASK:-1}"
echo "[sens] git sha=$(git rev-parse --short HEAD 2>/dev/null || echo none)"
# Default: the four agents that drive the ranking narrative (heuristic baseline,
# contextual bandit, PPO, Lagrangian PPO). Override via INFRARL_AGENTS.
: "${INFRARL_AGENTS:=heuristic,bandit_contextual,ppo,lagrangian_ppo}"
: "${INFRARL_OUT:=results_sensitivity_v1}"
: "${INFRARL_FACTORS:=0.8,0.9,1.1,1.2}"
python -m infrarl.evaluation.sensitivity \
  --agents "$INFRARL_AGENTS" \
  --seeds 3 \
  --factors "$INFRARL_FACTORS" \
  --axes quality,latency,cost \
  --out "$INFRARL_OUT"
echo "[sens] done at $(date)"
