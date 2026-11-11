#!/bin/bash
# One-time Zaratan env setup for InfraRL.
#
# Zaratan has no anaconda/miniconda/mamba module and ships Python 3.10 only.
# Our requirements.txt pins (numpy 2.5, scipy 1.18, pandas 3.0) need
# Python >= 3.11, so we install miniforge in user space and build a conda
# env at Python 3.11.
#
# Run from the repo root on a Zaratan login node:
#   bash scripts/zaratan/setup.sh

set -euo pipefail

CONDA_HOME="${CONDA_HOME:-$HOME/miniforge3}"
ENV_NAME="${INFRARL_ENV:-infrarl}"
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"

if ! command -v conda >/dev/null 2>&1 && [ ! -d "$CONDA_HOME" ]; then
  echo "[setup] no conda found, installing miniforge to $CONDA_HOME"
  # miniforge installer checks that $0 ends in .sh, so the temp file must too
  installer="$HOME/miniforge_installer.sh"
  curl -fsSL -o "$installer" "$MINIFORGE_URL"
  bash "$installer" -b -p "$CONDA_HOME"
  rm -f "$installer"
fi

# shellcheck disable=SC1091
source "$CONDA_HOME/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[setup] conda env '$ENV_NAME' already exists, reusing"
else
  echo "[setup] creating conda env '$ENV_NAME' with Python 3.11"
  conda create -n "$ENV_NAME" python=3.11 pip -y
fi

conda activate "$ENV_NAME"
pip install --upgrade pip
pip install -r requirements.txt

echo "[setup] smoke check: pytest + verify suite"
python -m pytest -q tests
python -m infrarl.evaluation.verify > /tmp/verify_$$.json
grep -c '"n_failed": 0' /tmp/verify_$$.json
rm -f /tmp/verify_$$.json

echo
echo "[setup] done. In a new shell, activate the env with:"
echo "    source $CONDA_HOME/etc/profile.d/conda.sh && conda activate $ENV_NAME"
echo "[setup] submit a run with:"
echo "    sbatch scripts/zaratan/run_experiment.sbatch"
