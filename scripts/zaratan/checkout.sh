#!/bin/bash
# Administrative operations for the HUMAN to run; no compute workloads or installs.
set -euo pipefail
[[ $# -ge 2 ]] || { echo 'usage: checkout.sh MODE PROJECT [--dry-run|RUN]' >&2; exit 2; }
mode="$1"
project="$2"
[[ "$project" == /* && -d "$project" ]] || exit 2
case "$mode" in
 inspect)
  git -C "$project" status --short
  git -C "$project" branch --show-current
  git -C "$project" rev-parse HEAD
  git -C "$project" remote -v
  ;;
 update)
  [[ "${3:-}" == '' || "${3:-}" == --dry-run ]] || exit 2
  [[ -z "$(git -C "$project" status --porcelain=v1 --untracked-files=all)" ]] || { echo 'Checkout is dirty (including untracked files). No update; inspect diffs/conflicts and reconcile manually. Never reset or overwrite local work.' >&2; exit 2; }
  [[ "$(git -C "$project" branch --show-current)" == main ]] || { echo 'Checkout must already be on main. No automatic branch switch.' >&2; exit 2; }
  for state in MERGE_HEAD rebase-merge rebase-apply CHERRY_PICK_HEAD REVERT_HEAD; do
   state_path="$(git -C "$project" rev-parse --path-format=absolute --git-path "$state")"
   [[ ! -e "$state_path" ]] || { echo 'Git operation in progress; update refused.' >&2; exit 2; }
  done
  remote="${INFRARL_GIT_REMOTE:-origin}"
  [[ "$remote" =~ ^[A-Za-z0-9_.-]+$ && "$remote" != -* ]] || exit 2
  if [[ "${3:-}" == --dry-run ]]; then
   printf 'Would fetch %s main, recheck clean tree, verify ancestry, then merge --ff-only FETCH_HEAD. No fetch/update performed.\n' "$remote"
   exit 0
  fi
  git -C "$project" fetch --no-tags "$remote" main
  [[ -z "$(git -C "$project" status --porcelain=v1 --untracked-files=all)" ]] || { echo 'Tree changed during fetch; no merge.' >&2; exit 2; }
  [[ "$(git -C "$project" branch --show-current)" == main ]] || { echo 'Branch changed during fetch; no merge.' >&2; exit 2; }
  git -C "$project" merge-base --is-ancestor HEAD FETCH_HEAD || { echo 'Branches diverged; no update. Resolve separately without discarding work.' >&2; exit 2; }
  git -C "$project" merge --ff-only --no-edit FETCH_HEAD
  ;;
 dependencies)
  bash "$(dirname -- "$0")/setup.sh" inspect
  ;;
 allocation)
  command -v sbatch
  sbalance
  sinfo -o '%P %a %l %D'
  squeue -u "$USER"
  if command -v scratch_quota >/dev/null; then scratch_quota; fi
  ;;
 monitor)
  run="${3:?absolute run directory required}"
  [[ "$run" == /* ]] || exit 2
  job="$(cat "$run/submitted-job-id.txt")"; job="${job%%;*}"
  [[ "$job" =~ ^[0-9]+$ ]] || { echo 'Invalid job ID' >&2; exit 2; }
  squeue -j "$job" || echo 'Job is no longer reported by squeue; inspect sacct.' >&2
  # Caller may redirect this accounting output to a NEW file outside artifacts/.
  sacct -j "$job" --parsable2 --format=JobID,State,ExitCode,Elapsed,MaxRSS
  ;;
 *) echo 'Unknown administrative operation' >&2; exit 2;;
esac
