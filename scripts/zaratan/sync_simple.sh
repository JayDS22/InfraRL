#!/bin/bash
# rsync helper: push code to Zaratan, pull results back.
#
#   bash scripts/zaratan/sync.sh push    # local repo -> Zaratan
#   bash scripts/zaratan/sync.sh pull    # Zaratan results/ -> local results/
#
# Set the remote path once:
#   export ZARATAN_HOST=login.zaratan.umd.edu
#   export ZARATAN_USER=<your-umd-directory-id>
#   export ZARATAN_PATH=/home/${ZARATAN_USER}/InfraRL

set -euo pipefail

HOST="${ZARATAN_HOST:-login.zaratan.umd.edu}"
USER="${ZARATAN_USER:?set ZARATAN_USER to your UMD directory ID}"
REMOTE="${ZARATAN_PATH:-/home/${USER}/InfraRL}"
LOCAL="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

EXCLUDES=(
  --exclude=".git/"
  --exclude="__pycache__/"
  --exclude=".pytest_cache/"
  --exclude=".venv/"
  --exclude="venv/"
  --exclude=".DS_Store"
  --exclude="*.pyc"
  --exclude="/results/"
  --exclude="/logs/"
  --exclude=".claude/"
)

case "${1:-}" in
  push)
    echo "[sync] push $LOCAL -> ${USER}@${HOST}:${REMOTE}"
    rsync -avz --delete "${EXCLUDES[@]}" "${LOCAL}/" "${USER}@${HOST}:${REMOTE}/"
    ;;
  pull)
    echo "[sync] pull ${USER}@${HOST}:${REMOTE}/results -> $LOCAL/results"
    rsync -avz "${USER}@${HOST}:${REMOTE}/results/" "${LOCAL}/results/"
    ;;
  *)
    echo "usage: $0 {push|pull}" >&2
    exit 2
    ;;
esac
