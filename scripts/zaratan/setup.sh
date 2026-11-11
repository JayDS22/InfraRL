#!/bin/bash
# One-time Zaratan env setup for InfraRL.
#
# Zaratan's system Python is 3.10, but our requirements.txt pins (numpy 2.5,
# scipy 1.18, pandas 3.0) need Python >= 3.11. Workaround: create a user-space
# conda env at Python 3.11 and install everything with pip.
#
# Run this once on a Zaratan login node from the repo root:
#   bash scripts/zaratan/setup.sh

set -euo pipefail

# Zaratan uses Lmod; anaconda3 is the standard conda module.
module purge
module load anaconda3 || module load conda

ENV_NAME="${INFRARL_ENV:-infrarl}"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[setup] conda env '$ENV_NAME' already exists, skipping create"
else
  echo "[setup] creating conda env '$ENV_NAME' with Python 3.11"
  conda create -n "$ENV_NAME" python=3.11 pip -y
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "[setup] installing pinned deps"
pip install --upgrade pip
pip install -r requirements.txt

echo "[setup] smoke check: pytest + verify suite"
python -m pytest -q tests
python -m infrarl.evaluation.verify > /tmp/verify_$$.json
grep '"n_failed": 0' /tmp/verify_$$.json | wc -l
rm -f /tmp/verify_$$.json

echo
echo "[setup] done. activate with:"
echo "    module load anaconda3 && conda activate $ENV_NAME"
echo "[setup] submit a run with:"
echo "    sbatch scripts/zaratan/run_experiment.sbatch"
