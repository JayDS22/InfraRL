#!/bin/bash
# Read-only inspection of the existing environment. No installs or changes.
set -euo pipefail
[[ "${1:-inspect}" == inspect ]] || { echo 'Only inspect is supported; environment changes are disabled.' >&2; exit 2; }
if [[ -z "${INFRARL_PYTHON:-}" ]]; then
 project="$(git -C "$(dirname -- "$0")/../.." rev-parse --show-toplevel)"
 if [[ -x "$project/.venv/bin/python" ]]; then INFRARL_PYTHON="$project/.venv/bin/python"
 elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then INFRARL_PYTHON="$CONDA_PREFIX/bin/python"
 else echo 'No existing environment detected; select INFRARL_PYTHON explicitly.' >&2; exit 2
 fi
fi
[[ "$INFRARL_PYTHON" == /* && -x "$INFRARL_PYTHON" ]] || exit 2
PYTHONDONTWRITEBYTECODE=1 "$INFRARL_PYTHON" - <<'PYINFO'
import importlib.metadata
import json
import sys
print(json.dumps({'executable': sys.executable, 'python': sys.version,
    'prefix': sys.prefix, 'base_prefix': sys.base_prefix,
    'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}}, indent=2))
PYINFO
echo 'Environment unchanged. Tests and training belong in approved Slurm jobs.'
