#!/bin/bash
# Never submits, deletes, or disables SSH host verification.
set -euo pipefail
mode="${1:-}"; [[ $# -gt 0 ]] && shift
preview=()
if [[ "${1:-}" == --dry-run ]]; then preview=(--dry-run); shift; fi
[[ $# == 2 ]] || { echo 'usage: sync_safe.sh {checkout|stage|controls|pull} [--dry-run] SOURCE DESTINATION' >&2; exit 2; }
# Deliberately constrain destinations to prevent shell/rsync option injection.
remote_pattern='^[A-Za-z0-9_.@-]+@login\.zaratan\.umd\.edu:/[A-Za-z0-9_./-]+$'
case "$mode" in
 checkout)
  root="$(realpath -- "$1")"
  [[ "$2" =~ $remote_pattern && "$2" != *'/../'* ]] || exit 2
  files="$root/scripts/zaratan/transfer_files.txt"
  : "${INFRARL_TRANSFER_BACKUP:?absolute approved REMOTE backup directory outside checkout required}"
  remote_path="${2#*:}"
  backup="$INFRARL_TRANSFER_BACKUP"
  [[ "$backup" == /* && "$backup" != "$remote_path" && "$backup" != "$remote_path/"* ]] || exit 2
  mkdir -p "$root/.workflow-validation"
  hashes="$(mktemp "$root/.workflow-validation/transfer_XXXXXXXX.sha256")"
  while IFS= read -r file; do
   [[ "$file" =~ ^[A-Za-z0-9_./-]+$ && "$file" != /* && "$file" != *'..'* ]] || exit 2
   case "$file" in .git/*|.venv*|hpc/*|zaratan_handoff/*|results/*) exit 2;; esac
   [[ -f "$root/$file" && ! -L "$root/$file" ]] || exit 2
   (cd "$root" && sha256sum -- "$file") >> "$hashes"
  done < "$files"
  backup="$backup/$(basename "$hashes" .sha256)"
  echo "Backup directory for this transfer: $backup"
  echo 'Review transfer_files.txt and remote git diff first. Unreviewed remote edits must not be overwritten.'
  rsync -rtvzc --itemize-changes --backup --backup-dir="$backup" "${preview[@]}" --files-from="$files" -- "$root/" "$2/"
  rsync -rtvzc --itemize-changes --ignore-existing "${preview[@]}" -- "$hashes" "$2/results/"
  echo "On Zaratan, from checkout root: sha256sum -c results/$(basename "$hashes")"
  ;;
 stage|controls)
  source_path="$(realpath -- "$1")"
  [[ "$2" =~ $remote_pattern && "$2" != *'/../'* && "$2" != *'/./'* ]] || exit 2
  if [[ "$mode" == stage ]]; then
   [[ -f "$source_path/snapshot-manifest.json" ]] || exit 2
   snapshot_hash="$(sha256sum "$source_path/snapshot-manifest.json")"; snapshot_hash="${snapshot_hash%% *}"
   PYTHONDONTWRITEBYTECODE=1 python3 "$source_path/scripts/zaratan/workflow.py" verify-snapshot --manifest "$source_path/snapshot-manifest.json" --expected-sha256 "$snapshot_hash"
   echo "Review and preserve manifest SHA256: $snapshot_hash"
  else
   [[ -f "$source_path/approval.json" && -f "$source_path/submit.sh" && -f "$source_path/run-info.json" ]] || exit 2
  fi
  if [[ ${#preview[@]} == 0 ]]; then
   # Destination MUST be fresh. Parent path must already be authorized/created.
   remote_host="${2%%:*}"; remote_path="${2#*:}"
   ssh "$remote_host" "mkdir -- '$remote_path'"
  fi
  rsync -rlptvz --itemize-changes "${preview[@]}" -- "$source_path/" "$2/"
  if [[ "$mode" == stage && ${#preview[@]} == 0 ]]; then
   ssh "$remote_host" "cd '$remote_path' && PYTHONDONTWRITEBYTECODE=1 python3 scripts/zaratan/workflow.py verify-snapshot --manifest snapshot-manifest.json --expected-sha256 '$snapshot_hash'"
  fi
  ;;
 pull)
  [[ "$1" =~ $remote_pattern && "$1" != *'/../'* && "$2" != -* ]] || exit 2
  if [[ ${#preview[@]} == 0 ]]; then
   [[ ! -e "$2" ]] || { echo 'Pull destination must be new.' >&2; exit 2; }
   mkdir -p -- "$2"
  fi
  rsync -rlptvz --itemize-changes "${preview[@]}" -- "$1/" "$2/"
  ;;
 *) echo 'Use checkout for reviewed file overlays, or pull for results; snapshot modes remain optional.' >&2; exit 2;;
esac
