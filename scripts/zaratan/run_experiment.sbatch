#!/bin/bash
# HUMAN submits from the existing checkout; logs must exist before sbatch.
#SBATCH --job-name=infrarl
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Slurm compute allocation required}"
[[ $# -ge 1 ]] || { echo 'usage: sbatch [approved resources] run_experiment.sbatch PHASE [options]' >&2; exit 2; }
export PYTHONDONTWRITEBYTECODE=1 INFRARL_ON_CLUSTER=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export TOKENIZERS_PARALLELISM=false
# Retain the earlier optional final-snapshot interface, without requiring it.
if [[ "$1" == *.json ]]; then
 [[ $# == 3 ]] || exit 2
 mapfile -t settings < <(python3 - "$1" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
print(c['python']); print(c['remote_source']); print(c['phase'])
PY
 )
 [[ ${#settings[@]} == 3 ]] || exit 2
 export MPLCONFIGDIR="${INFRARL_RUNTIME_CACHE:?}/mpl-${SLURM_JOB_ID}"
 cd "${settings[1]}"
 exec "${settings[0]}" scripts/zaratan/workflow.py execute --config "$1" --approval "$2" --expected-sha256 "$3" --phase "${settings[2]}"
fi
[[ "${INFRARL_RESOURCES_APPROVED:-}" == yes ]] || { echo 'Resource approval must precede submission.' >&2; exit 2; }
project="$(git -C "${SLURM_SUBMIT_DIR:?}" rev-parse --show-toplevel)"
cd "$project"
if [[ -n "${INFRARL_PYTHON:-}" ]]; then
 interpreter="$INFRARL_PYTHON"
elif [[ -x "$project/.venv/bin/python" ]]; then
 interpreter="$project/.venv/bin/python"
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
 interpreter="$CONDA_PREFIX/bin/python"
else
 echo 'Select an existing .venv or activated Conda interpreter; no installation is performed.' >&2; exit 2
fi
[[ "$interpreter" == /* && -x "$interpreter" ]] || exit 2
export INFRARL_RUNTIME_CACHE="$project/results/${1}_${SLURM_JOB_ID}/runtime-cache"
export MPLCONFIGDIR="$INFRARL_RUNTIME_CACHE/matplotlib"
mkdir -p "$INFRARL_RUNTIME_CACHE"
exec "$interpreter" scripts/zaratan/workflow.py checkout "$@"
