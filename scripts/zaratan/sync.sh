#!/bin/bash
# Legacy mirroring is disabled. Never delete remote files.
set -euo pipefail
echo 'Legacy sync disabled. Use sync_safe.sh stage/pull with an explicit isolated destination.' >&2
exit 2
