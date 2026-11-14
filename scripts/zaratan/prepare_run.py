#!/usr/bin/env python3
"""Prepare a local handoff bundle. Never connects, installs packages or submits."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import uuid

import workflow as w




def controls(c, manifest_sha, destination, prerequisites):
    """Write approval (false by default) and a submission script for the human."""
    destination.mkdir(parents=True, exist_ok=False)
    approval = {'config_sha256': w.canonical(c), 'snapshot_manifest_sha256': manifest_sha,
                'resources_approved': False, 'provisional_approved': False,
                'scientific_decisions_approved': False, 'approved_by': '', 'approved_utc': '',
                'prerequisite_results': [{'path': path, 'sacct': sacct} for path, sacct in prerequisites]}
    w.save(destination / 'approval.json', approval)
    source = Path(c['remote_source'])
    run = Path(c['remote_results']) / c['run_id']
    resources = c['resources']
    command = ['sbatch', '--parsable', '--job-name=infrarl', '--account=' + c['account'],
               '--partition=standard', '--nodes=1', '--ntasks=1',
               '--cpus-per-task=' + str(resources['cpus']), '--mem=' + resources['memory'],
               '--time=' + resources['time'], '--chdir=' + str(source),
               '--output=' + str(run / 'logs/%x-%j.out'), '--error=' + str(run / 'logs/%x-%j.err'),
               str(source / 'scripts/zaratan/run_experiment.sbatch'), str(source / 'run-config.json'),
               str(run / 'approval.json'), manifest_sha]
    # No permission is inferred from generating this file. A human must record the
    # actual resource approval and execute this script in their remote terminal.
    script = '#!/bin/bash\nset -euo pipefail\n'
    script += 'case "${1:-}" in ""|--dry-run) ;; *) echo "usage: submit.sh [--dry-run]" >&2; exit 2;; esac\n'
    script += 'if [[ "${1:-}" == --dry-run ]]; then\n'
    script += "  printf '%s\\n' " + shlex.quote(shlex.join(command)) + '\n  exit 0\nfi\n'
    script += 'RUN=' + shlex.quote(str(run)) + '\n'
    script += 'SOURCE=' + shlex.quote(str(source)) + '\n'
    script += "python3 - \"$RUN/approval.json\" <<'CHECK_APPROVAL'\n"
    script += 'import json,sys\na=json.load(open(sys.argv[1]))\n'
    script += 'if a.get("resources_approved") is not True: sys.exit("Resource approval has not been recorded")\n'
    script += 'if not (a.get("approved_by") and a.get("approved_utc")): sys.exit("Human approval provenance required")\n'
    script += 'if a.get("config_sha256") != ' + repr(w.canonical(c)) + ': sys.exit("Config approval mismatch")\n'
    script += 'if a.get("snapshot_manifest_sha256") != ' + repr(manifest_sha) + ': sys.exit("Snapshot approval mismatch")\nCHECK_APPROVAL\n'
    script += '[[ ! -e "$RUN/submitted-job-id.txt" && ! -e "$RUN/artifacts" ]] || { echo "Run already submitted/executed; prepare a fresh run." >&2; exit 2; }\n'
    script += 'PYTHONDONTWRITEBYTECODE=1 python3 "$SOURCE/scripts/zaratan/workflow.py" verify-snapshot --manifest "$SOURCE/snapshot-manifest.json" --expected-sha256 ' + shlex.quote(manifest_sha) + '\n'
    script += 'mkdir -p "$RUN/logs" "$RUN/runtime-cache"\nexport INFRARL_RUNTIME_CACHE="$RUN/runtime-cache"\n'
    script += 'job_id=(' + shlex.join(command) + ')\n'
    script += '"${job_id[@]}" > "$RUN/submitted-job-id.txt"\ncat "$RUN/submitted-job-id.txt"\n'
    (destination / 'submit.sh').write_text(script)
    w.save(destination / 'run-info.json', {'config': c, 'snapshot_manifest_sha256': manifest_sha})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=w.PHASES, required=True)
    p.add_argument('--project', default=os.environ.get('INFRARL_PROJECT'), help='authorized absolute remote checkout path')
    p.add_argument('--work-root', default=os.environ.get('INFRARL_WORK_ROOT'), help='authorized staging/results directory outside the checkout')
    p.add_argument('--account', default=os.environ.get('INFRARL_ACCOUNT'), help='authorized Slurm account')
    p.add_argument('--python', help='existing remote interpreter, defaults to PROJECT/.venv/bin/python')
    p.add_argument('--destination', required=True, help='new local bundle directory')
    p.add_argument('--prerequisite', action='append', nargs=2, default=[], metavar=('ARTIFACTS_PATH', 'SACCT_PATH'))
    p.add_argument('--hyperparameters-frozen', action='store_true')
    p.add_argument('--scenario', help='JSON file, required for approved sensitivity scenario')
    p.add_argument('--ablate', default='ppo')
    args = p.parse_args()
    if not args.project or not args.work_root or not Path(args.project).is_absolute() or not Path(args.work_root).is_absolute() or not args.account:
        p.error('--project, --work-root and --account (or INFRARL_* variables) are required')
    if Path(args.work_root).is_relative_to(Path(args.project)):
        p.error('work-root must be outside checkout to keep Git update checks clean')
    c = w.load(w.ROOT / 'scripts/zaratan/checks_config.json')
    c['account'] = args.account
    c['remote_results'] = args.work_root.rstrip('/') + '/results'
    c['remote_checkout'] = args.project
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    c['run_id'] = args.phase + '_' + stamp + '_' + uuid.uuid4().hex[:6]
    c['phase'] = args.phase
    c['remote_source'] = args.work_root.rstrip('/') + '/snapshots/' + c['run_id']
    c['python'] = args.python or args.project.rstrip('/') + '/.venv/bin/python'
    c['python_version'] = '3.13'
    c['hyperparameters_frozen'] = args.hyperparameters_frozen
    if args.phase in ('functional-smoke', 'ppo-update-smoke'):
        c.update(n_train=50, n_validation=50, n_test=50, seeds=1, epochs=1)
        c['resources']['time'] = '00:15:00'
    if args.phase == 'calibration':
        c['resources'].update(cpus=1, memory='4G', time='00:15:00')
    if args.phase == 'ablations':
        c['ablate'] = args.ablate
    if args.phase == 'sensitivity':
        if not args.scenario:
            p.error('sensitivity requires an explicit approved scenario JSON')
        c['scenario'] = w.load(args.scenario)
    if args.phase in ('frozen', 'ablations') and not args.hyperparameters_frozen:
        p.error('freeze hyperparameters before preparing frozen/ablation runs')
    bundle = Path(args.destination).resolve()
    bundle.mkdir(parents=True, exist_ok=False)
    w.save(bundle / 'config.json', c)
    w.config(bundle / 'config.json')
    w.snapshot(c, bundle / 'source')
    sha = w.digest(bundle / 'source/snapshot-manifest.json')
    controls(c, sha, bundle / 'control', args.prerequisite)
    print('Prepared bundle:', bundle)
    print('Remote source:', c['remote_source'])
    print('Remote controls/results:', c['remote_results'] + '/' + c['run_id'])
    print('Approval flags are false. No transfer, environment change or submission performed.')


if __name__ == '__main__':
    main()
