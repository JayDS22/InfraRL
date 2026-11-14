#!/usr/bin/env python3
"""CPU workflow. Planning and packaging never submit jobs. Stdlib-only until execute."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
PHASES = ('checks', 'calibration', 'functional-smoke', 'ppo-update-smoke',
          'development', 'frozen', 'ablations', 'sensitivity', 'final-checks')
OUTCOMES = {'passed', 'failed', 'infeasible', 'unverified', 'incomparable'}
SOURCE_DIRS = ('infrarl', 'tests', 'scripts/zaratan', 'data/benchmarks')
SOURCE_FILES = ('README.md', 'scripts/zaratan/transfer_files.txt', 'requirements.txt',
                'pyproject.toml', 'uv.lock', 'scripts/run_calibration.py',
                'scripts/generate_profile_exemplars.py', 'docs/EVALUATION.md')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def load(path):
    return json.loads(Path(path).read_text())


def need(ok, message):
    if not ok:
        raise ValueError(message)


def config(path):
    return validate_config(load(path))


def validate_config(c):
    need(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', c['run_id']), 'invalid run_id')
    need(c['phase'] in PHASES, 'unknown phase')
    need(isinstance(c['agents'], list) and c['agents'] and len(set(c['agents'])) == len(c['agents']), 'explicit unique agents required')
    need(all(re.fullmatch(r'[a-z_]+', a) for a in c['agents']), 'invalid agent name')
    r = c['resources']
    need(r['partition'] == 'standard' and r.get('gpus', 0) == 0, 'CPU standard partition only')
    need(isinstance(r['cpus'], int) and r['cpus'] > 0, 'positive CPUs required')
    need(re.fullmatch(r'[1-9][0-9]*[GM]', r['memory']), 'memory must be explicit G/M')
    need(re.fullmatch(r'[0-9]{2,3}:[0-5][0-9]:[0-5][0-9]', r['time']), 'explicit HH:MM:SS required')
    need(re.fullmatch(r'[A-Za-z0-9_-]+', c['account']), 'authorized account required')
    for name in ('remote_source', 'remote_results', 'python'):
        need(Path(c[name]).is_absolute() and not any(x.isspace() for x in c[name]), name + ' must be absolute without whitespace')
    if c.get('source_mode') != 'checkout':
        need(not (Path(c['remote_results']) / c['run_id']).resolve().is_relative_to(Path(c['remote_source']).resolve()), 'results must be outside source')
    for k in ('n_train', 'n_validation', 'n_test', 'seeds'):
        need(isinstance(c[k], int) and c[k] > 0, k + ' must be positive')
    need(c['seeds'] <= 1000, 'training seed overlap')
    need(c.get('epochs') is None or isinstance(c['epochs'], int) and c['epochs'] > 0, 'epochs must be positive')
    need(c['calibration']['outcome'] in OUTCOMES, 'invalid calibration outcome')
    if c['phase'] in ('frozen', 'ablations'):
        need(c['seeds'] >= 3 and c['n_train'] == c['n_test'] == 1000, 'frozen/ablation requires >=3 seeds and 1000 profiles')
        need(c.get('hyperparameters_frozen') is True, 'freeze hyperparameters first')
    if c['phase'] == 'functional-smoke':
        need((c['n_train'], c['n_validation'], c['seeds'], c['epochs']) == (50, 50, 1, 1), 'functional smoke uses 50/50 profiles, one seed/epoch')
    if c['phase'] == 'ablations':
        need(c.get('ablate') in c['agents'], 'explicit ablation agent required')
    if c['phase'] == 'sensitivity':
        s = c.get('scenario', {})
        need(s and set(s) <= {'reward_weights', 'accuracy_range'}, 'explicit supported scenario required')
        if 'reward_weights' in s:
            w = s['reward_weights']
            need(len(w) == 3 and all(math.isfinite(v) and v >= 0 for v in w) and abs(sum(w)-1) < 1e-9, 'invalid reward weights')
        if 'accuracy_range' in s:
            lo, hi = s['accuracy_range']
            need(0 < lo < hi <= 1, 'invalid accuracy range')
    elif c.get('scenario'):
        raise ValueError('scenario allowed only in sensitivity')
    return c


def source_files(root):
    selected = set()
    for d in SOURCE_DIRS:
        if (root / d).exists():
            for p in (root / d).rglob('*'):
                if '__pycache__' in p.parts or '.pytest_cache' in p.parts:
                    continue
                if p.is_file():
                    # Positive extension list, including explicitly staged untracked code.
                    if p.suffix in ('.py', '.sh', '.sbatch', '.md', '.csv', '.json'):
                        need(not any(word in p.name.lower() for word in ('credential', 'secret', 'private_key', 'token')), 'sensitive filename rejected')
                        need(not p.is_symlink() and not any(x.startswith('.') for x in p.relative_to(root).parts), 'hidden/symlink source rejected')
                        selected.add(p.relative_to(root).as_posix())
    for name in SOURCE_FILES:
        if (root / name).is_file():
            need(not (root / name).is_symlink(), 'symlink source rejected')
            selected.add(name)
    return sorted(selected)


def snapshot(c, destination, root=ROOT):
    dest = Path(destination).resolve()
    dest.mkdir(parents=True, exist_ok=False)
    names = source_files(root)
    hashes = {}
    for name in names:
        data = (root / name).read_bytes()
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
        target.chmod(0o444)
    git = lambda *args: subprocess.check_output(['git', '-C', str(root), *args])
    tracked = set(git('ls-files').decode().splitlines())
    save(dest / 'run-config.json', c)
    hashes['run-config.json'] = digest(dest / 'run-config.json')
    manifest = {'schema': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'head_sha': git('rev-parse', 'HEAD').decode().strip(),
                'dirty_diff_sha256': hashlib.sha256(git('diff', '--binary', 'HEAD')).hexdigest(),
                'included_untracked': sorted(set(names) - tracked),
                'files': hashes, 'config_sha256': canonical(c)}
    save(dest / 'snapshot-manifest.json', manifest)
    (dest / 'run-config.json').chmod(0o444)
    (dest / 'snapshot-manifest.json').chmod(0o444)
    print(json.dumps({'snapshot': str(dest), 'manifest_sha256': digest(dest / 'snapshot-manifest.json'), 'files': len(hashes)}, indent=2))


def verify_snapshot(path, expected):
    path = Path(path).resolve()
    need(digest(path) == expected, 'snapshot manifest hash mismatch')
    m = load(path)
    root = path.parent
    need(not any(p.is_symlink() for p in root.rglob('*')), 'symlinks prohibited in snapshot')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    need(actual == set(m['files']) | {path.name}, 'missing or unexpected snapshot files')
    for name, value in m['files'].items():
        p = root / name
        need(not p.is_symlink() and p.resolve().is_relative_to(root), 'unsafe source path')
        need(digest(p) == value, 'source hash mismatch: ' + name)
    need(canonical(load(root / 'run-config.json')) == m['config_sha256'], 'config mismatch')
    return m


def plan(c):
    r = c['resources']
    source, results = Path(c['remote_source']), Path(c['remote_results']) / c['run_id']
    args = ['sbatch', '--account=' + c['account'], '--partition=standard', '--nodes=1', '--ntasks=1',
            '--cpus-per-task=' + str(r['cpus']), '--mem=' + r['memory'], '--time=' + r['time'],
            '--chdir=' + str(source), '--output=' + str(results / 'logs/%x-%j.out'),
            '--error=' + str(results / 'logs/%x-%j.err'), str(source / 'scripts/zaratan/run_experiment.sbatch'),
            str(source / 'run-config.json'), '<APPROVAL_JSON>', '<SNAPSHOT_MANIFEST_SHA256>']
    print('PLAN ONLY: no directories created remotely; no submission performed.')
    print('Before submission: create fresh run parent and logs:', results / 'logs')
    print(shlex.join(args))
    print('Approval record (human attestation; not an automatic authorization mechanism):')
    print(json.dumps({'config_sha256': canonical(c), 'snapshot_manifest_sha256': '<FROM SNAPSHOT>',
                      'resources_approved': False, 'provisional_approved': False,
                      'scientific_decisions_approved': False, 'approved_by': '<HUMAN>',
                      'approved_utc': '<UTC>', 'prerequisite_results': []}, indent=2))


def dependency_report(requirements, installed):
    normalize = lambda name: re.sub(r'[-_.]+', '-', name).lower()
    versions = {normalize(k): v for k, v in installed.items()}
    rows = []
    for line in Path(requirements).read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)==([^ ;]+)', line)
        need(match, 'only exact dependency pins supported; review requirements')
        name, expected = match.groups()
        actual = versions.get(normalize(name))
        rows.append({'package': name, 'expected': expected, 'installed': actual, 'matches': actual is not None and actual.split('+', 1)[0] == expected})
    return rows


def inventory(names):
    from infrarl.agents.registry import REGISTRY
    need(all(a in REGISTRY for a in names), f'unavailable agents: {set(names) - set(REGISTRY)}')
    return {a: {'train_epochs': REGISTRY[a].train_epochs,
                'factory': inspect.getsource(REGISTRY[a].factory).strip()} for a in names}


def ppo_smoke(out):
    import torch
    from infrarl.agents.ppo_agent import PPOAgent
    from infrarl.evaluation.experiments import run_episode
    from infrarl.simulator.environment import InfraEnv
    from infrarl.simulator.profiles import generate_profiles
    agent = PPOAgent(seed=0, rollout_size=32, minibatch_size=16, update_epochs=1)
    before = [p.detach().clone() for p in agent.net.parameters()]
    count = [0]
    step = agent.opt.step
    def counted(*args, **kwargs):
        count[0] += 1
        return step(*args, **kwargs)
    agent.opt.step = counted
    profiles = generate_profiles(20, 4000)
    env = InfraEnv(profiles)
    for p in profiles:
        run_episode(env, agent, p, train=True)
    changed = any(not torch.equal(a, b) for a, b in zip(before, agent.net.parameters()))
    finite = all(bool(torch.isfinite(p).all()) for p in agent.net.parameters())
    need(count[0] > 0 and changed and finite, 'PPO optimizer did not update finite parameters')
    need(all(p.device.type == 'cpu' for p in agent.net.parameters()), 'PPO must run on CPU')
    save(out / 'ppo-update.json', {'optimizer_steps': count[0], 'parameters_changed': changed,
                                 'finite': finite, 'rollout_size': 32, 'profile_seed': 4000,
                                 'profiles': 20, 'purpose': 'optimizer execution, not convergence'})


def experiment(c, out):
    from infrarl.evaluation import run_experiment as driver
    split = 'test' if c['phase'] in ('frozen', 'ablations') else 'validation'
    if c['phase'] == 'sensitivity':
        # Isolated process only. Never mutate canonical source or baseline artifacts.
        import dataclasses
        import numpy as np
        import infrarl.simulator.environment as envmod
        s = c['scenario']
        if 'reward_weights' in s:
            envmod.REWARD_WEIGHTS = tuple(s['reward_weights'])
        if 'accuracy_range' in s:
            original = driver.generate_profiles
            def generate(n, seed):
                rng = np.random.default_rng(seed + 10000)
                return [dataclasses.replace(p, acc_req=float(rng.uniform(*s['accuracy_range']))) for p in original(n, seed)]
            driver.generate_profiles = generate
    args = ['run_experiment', '--agents', ','.join(c['agents']), '--n-train', str(c['n_train']),
            '--n-test', str(c['n_test']), '--n-validation', str(c['n_validation']),
            '--seeds', str(c['seeds']), '--evaluation-split', split,
            '--ablate', c.get('ablate', '') if c['phase'] == 'ablations' else '', '--out', str(out / 'experiment')]
    if c.get('epochs') is not None:
        args += ['--epochs', str(c['epochs'])]
    sys.argv = args
    driver.main()


def file_hashes(root):
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob('*')) if p.is_file() and p.relative_to(root).as_posix() != 'manifest.json'}


def validate_artifacts(out):
    out = Path(out)
    m = load(out / 'manifest.json')
    need(m['exit_code'] == 0 and m['status'] == 'execution-complete', 'execution failed/incomplete')
    need(file_hashes(out) == m['artifact_sha256'], 'result hashes differ')
    c = m['config']
    need(m['config_sha256'] == canonical(c), 'manifest config changed')
    required = {'checks': ['verification.json', 'commands.log'], 'final-checks': ['verification.json', 'commands.log'],
                'calibration': ['calibration.json'], 'ppo-update-smoke': ['ppo-update.json']}
    for name in required.get(c['phase'], ['experiment/episodes.csv']):
        need((out / name).is_file() and (out / name).stat().st_size > 0, 'missing phase artifact: ' + name)
    if c['phase'] in ('checks', 'final-checks'):
        v = load(out / 'verification.json')
        need(all(v[k]['n_failed'] == 0 for k in ('masks', 'simulator', 'reward', 'determinism')), 'verification failures')
        with (out / 'exemplars/profile_exemplars.csv').open() as f:
            exemplars = list(csv.DictReader(f))
        expected_use_cases = {'chatbot', 'code_generation', 'batch_inference', 'rag', 'agentic', 'structured_extraction'}
        need(len(exemplars) == 6 and {r['use_case'] for r in exemplars} == expected_use_cases, 'exemplar coverage mismatch')
        need(len({r['profile_index'] for r in exemplars}) == 6, 'duplicate exemplar profiles')
        need((out / 'exemplars/PROFILE_EXEMPLARS.md').stat().st_size > 0, 'missing exemplar method metadata')
    if c['phase'] == 'calibration':
        report = load(out / 'calibration.json')
        need(report['outcome'] in OUTCOMES, 'invalid calibration outcome')
        need(load(out / 'calibration/summary.json') == report, 'calibration report mismatch')
        for name in ('residuals.csv', 'report.md'):
            need((out / 'calibration' / name).stat().st_size > 0, 'missing calibration artifact')
    if c['phase'] == 'ppo-update-smoke':
        v = load(out / 'ppo-update.json')
        need(v['optimizer_steps'] > 0 and v['finite'] and v['parameters_changed'], 'optimizer smoke failed')
    if (out / 'experiment').exists():
        e = out / 'experiment'
        def rows(name):
            path = e / name
            need(path.is_file() and path.stat().st_size > 0, 'missing/empty artifact: ' + name)
            with path.open() as handle:
                return list(csv.DictReader(handle))
        episodes = rows('episodes.csv')
        summary = rows('summary_per_seed.csv')
        aggregate = rows('summary.csv')
        curves = rows('curves.csv')
        rows('stats.csv')  # Header-only is valid when fewer than two agents are compared.
        exp = load(e / 'experiment.json')
        metrics = ('avg_quality', 'quality_attainment', 'quality_compliance', 'budget_compliance',
                   'sla_compliance', 'meets_all', 'mean_reward', 'mean_cost_usd')
        for name in ('figures/pareto_quality_vs_cost.png', 'figures/heatmap_meets_all_by_use_case.png', 'figures/compliance.png'):
            need((e / name).is_file() and (e / name).stat().st_size > 0, 'missing/empty figure')
        n = c['n_test'] if c['phase'] in ('frozen', 'ablations') else c['n_validation']
        expected_keys = {(a, s, i) for a in c['agents'] for s in range(c['seeds']) for i in range(n)}
        actual_keys = [(r['agent'], int(r['seed']), int(r['profile_idx'])) for r in episodes]
        need(len(actual_keys) == len(expected_keys) and set(actual_keys) == expected_keys, 'episode count/coverage mismatch')
        totals = {(a, s): 0 for a in c['agents'] for s in range(c['seeds'])}
        for row in episodes:
            values = {k: float(row[k]) for k in ('quality', 'latency_ms', 'cost_usd', 'reward', 'acc_req', 'budget', 'sla_ms')}
            need(all(math.isfinite(v) for v in values.values()), 'nonfinite episode metric')
            need(0 <= values['quality'] <= 1 and values['cost_usd'] >= 0 and values['latency_ms'] >= 0, 'invalid episode metrics')
            need(0 < values['acc_req'] <= 1 and values['budget'] > 0 and values['sla_ms'] > 0, 'invalid profile metrics')
            need(0 <= values['reward'] <= float(m['reward']['maximum']) + 1e-4, 'reward out of bounds')
            flags = [row[k] for k in ('within_budget', 'meets_sla', 'meets_quality')]
            need(all(v in ('0', '1') for v in flags), 'invalid compliance flags')
            totals[row['agent'], int(row['seed'])] += all(v == '1' for v in flags)
        keys = [(r['agent'], int(r['seed'])) for r in summary]
        need(len(keys) == len(totals) and set(keys) == set(totals), 'summary count/coverage mismatch')
        for row in summary:
            for col in metrics:
                value = float(row[col])
                need(math.isfinite(value), 'invalid summary metric ' + col)
                if col not in ('mean_reward', 'mean_cost_usd'):
                    need(0 <= value <= 1, 'summary metric out of bounds')
            need(abs(totals[row['agent'], int(row['seed'])]/n - float(row['meets_all'])) < 1e-9, 'joint compliance mismatch')
        need(len(aggregate) == len(c['agents']) and {r['agent'] for r in aggregate} == set(c['agents']), 'aggregate inventory mismatch')
        need(exp['curve_split'] == 'validation' and exp['validation_seed'] == 3000 and exp['test_seed'] == 1000, 'split provenance mismatch')
        need(exp['agents'].split(',') == c['agents'], 'experiment inventory mismatch')
        expected_split = 'test' if c['phase'] in ('frozen', 'ablations') else 'validation'
        need(exp['evaluation_split'] == expected_split, 'evaluation split mismatch')
        curve_keys = [(r['agent'], int(r['seed']), int(r['epoch'])) for r in curves]
        expected_curves = {(a, s, epoch) for a in c['agents'] for s in range(c['seeds'])
                           for epoch in range(1, exp['budgets'][a]['epochs']+1)}
        need(len(curve_keys) == len(expected_curves) and set(curve_keys) == expected_curves, 'curve count/coverage mismatch')
        need(all(math.isfinite(float(r[k])) for r in curves for k in ('train_reward', 'eval_reward')), 'invalid curves')
        if curves:
            need((e / 'figures/learning_curves.png').stat().st_size > 0, 'missing curve figure')
        if c['phase'] == 'ablations':
            abl = rows('ablations.csv')
            variants = {'full', 'no_hardware_awareness', 'no_precision_tuning', 'no_scaling_policy', 'no_multitier'}
            keys = [(r['variant'], int(r['seed'])) for r in abl]
            need(len(keys) == len(variants)*c['seeds'] and set(keys) == {(v, s) for v in variants for s in range(c['seeds'])}, 'ablation count/coverage mismatch')
            need(all(r['agent'] == c['ablate'] and all(math.isfinite(float(r[k])) for k in metrics) for r in abl), 'invalid ablation metrics')
            need((e / 'figures/ablations.png').stat().st_size > 0, 'missing ablation figure')
    return m


def validate_completed(out, accounting):
    m = validate_artifacts(out)
    rows = list(csv.DictReader(Path(accounting).read_text().splitlines(), delimiter='|'))
    job = [r for r in rows if r.get('JobID') == m['slurm']['SLURM_JOB_ID']]
    need(len(job) == 1 and job[0]['State'].strip() == 'COMPLETED' and job[0]['ExitCode'].strip() == '0:0', 'sacct job not COMPLETED/0:0')
    steps = [r for r in rows if r.get('JobID', '').startswith(m['slurm']['SLURM_JOB_ID'] + '.')]
    need(all(r['State'].strip() == 'COMPLETED' and r['ExitCode'].strip() == '0:0' for r in steps), 'failed Slurm step')
    name = 'infrarl-' + m['slurm']['SLURM_JOB_ID'] + '.out'
    log = Path(out).parent / 'logs' / name
    if not log.exists(): log = Path(out).parent.parent / 'logs' / name
    need(log.is_file() and log.stat().st_size > 0, 'missing/empty Slurm stdout log')
    return m


def validate_calibration_evidence(evidence):
    need(evidence.get('complete') is True and evidence.get('outcome') == 'passed' and evidence.get('matched_references') is True, 'incomplete calibration evidence')
    required = evidence.get('required_cells', [])
    cells = evidence.get('cells', [])
    need(required and len(set(required)) == len(required), 'declared calibration cells required')
    need(len(cells) == len(required) and {r['id'] for r in cells} == set(required), 'calibration coverage incomplete')
    for row in cells:
        need(row['outcome'] == 'passed' and row['comparable'] is True and row['independent_reference'] is True, 'unmatched calibration cell')
        ref, prediction = float(row['reference']), float(row['prediction'])
        need(math.isfinite(ref) and ref > 0 and math.isfinite(prediction), 'invalid calibration numbers')
        need(100 * abs(prediction-ref) / abs(ref) <= 15, 'calibration residual failed')
        for field in ('source_url', 'retrieved_utc', 'model_id', 'hardware', 'precision', 'workload', 'metric', 'units'):
            need(row.get(field), 'missing calibration provenance: ' + field)


def checkout_source(c):
    git = lambda *args: subprocess.check_output(['git', '-C', str(ROOT), *args])
    names = source_files(ROOT)
    tracked = set(git('ls-files').decode().splitlines())
    return {'head_sha': git('rev-parse', 'HEAD').decode().strip(),
            'dirty_diff_sha256': hashlib.sha256(git('diff', '--binary', 'HEAD')).hexdigest(),
            'included_untracked': sorted(set(names)-tracked),
            'files': {name: digest(ROOT/name) for name in names}, 'config_sha256': canonical(c)}


def verify_checkout(c, source):
    current = checkout_source(c)
    need(current == source, 'Checkout changed during job; preserve outputs and review before reporting')


def archive_frozen_source(source, c, out):
    target=Path(out)/'frozen_source.tar'
    with tarfile.open(target,'x') as archive:
        contents={name:(ROOT/name).read_bytes() for name in source['files']}
        for name,data in contents.items():
            need(hashlib.sha256(data).hexdigest()==source['files'][name], 'source changed before frozen archive')
        contents['__run_config.json']=(json.dumps(c,sort_keys=True,indent=2)+'\n').encode()
        contents['__source_manifest.json']=(json.dumps(source,sort_keys=True,indent=2)+'\n').encode()
        for name,data in contents.items():
            info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o444;info.mtime=0
            archive.addfile(info,io.BytesIO(data))
    target.chmod(0o444)


def checkout_phase(args):
    need(os.environ.get('SLURM_JOB_ID'), 'compute phases require Slurm allocation')
    need(os.environ.get('INFRARL_RESOURCES_APPROVED') == 'yes', 'record resource approval before sbatch')
    c = load(ROOT/'scripts/zaratan/checks_config.json')
    job = os.environ['SLURM_JOB_ID']
    record = subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
    limit = re.search(r'\bTimeLimit=(\S+)',record)
    need(limit, 'Slurm walltime unavailable')
    days, clock = limit[1].split('-') if '-' in limit[1] else ('0', limit[1])
    parts = list(map(int,clock.split(':')))
    if len(parts)==2: parts.insert(0,0)
    time = f'{int(days)*24+parts[0]:02d}:{parts[1]:02d}:{parts[2]:02d}'
    c.update(phase=args.phase, run_id=f'{args.phase}_{job}', source_mode='checkout',
             remote_source=str(ROOT), remote_results=str(ROOT/'results'), python=sys.executable,
             python_version=f'{sys.version_info.major}.{sys.version_info.minor}',
             account=os.environ.get('SLURM_JOB_ACCOUNT',''),
             resources={'partition':'standard','cpus':int(os.environ.get('SLURM_CPUS_PER_TASK','0')),
                        'memory':os.environ.get('SLURM_MEM_PER_NODE','0')+'M','time':time,'gpus':0},
             hyperparameters_frozen=args.hyperparameters_frozen)
    if args.agents is not None: c['agents']=args.agents.split(',')
    if args.phase=='functional-smoke': c.update(n_train=50,n_validation=50,n_test=50,seeds=1,epochs=1)
    if args.epochs is not None: c['epochs']=args.epochs
    if args.seeds is not None: c['seeds']=args.seeds
    if args.phase=='ablations': c['ablate']=args.ablate
    if args.calibration_report:
        report=Path(args.calibration_report).resolve()
        need(report.is_relative_to(ROOT/'data/benchmarks'), 'reviewed calibration evidence must be under data/benchmarks')
        evidence=load(report);validate_calibration_evidence(evidence)
        c['calibration']={'outcome':'passed','report':report.relative_to(ROOT).as_posix(),'sha256':digest(report)}
    approval={'resources_approved':True,'provisional_approved':args.provisional,
              'scientific_decisions_approved':args.hyperparameters_frozen or args.phase=='development',
              'approved_by':os.environ.get('USER','manual-user'),'approved_utc':datetime.now(timezone.utc).isoformat(),
              'config_sha256':canonical(c),'prerequisite_results':[]}
    # Same phase executor and validators; no source copy/control bundle for development.
    execute(c, approval, None)


def execute(c, approval_path, expected):
    validate_config(c)
    need(sys.dont_write_bytecode, 'set PYTHONDONTWRITEBYTECODE=1 before compute execution')
    need(os.environ.get('SLURM_JOB_ID'), 'compute phases require Slurm allocation')
    expected_python = c.get('python_version', '3.13')
    need(f'{sys.version_info.major}.{sys.version_info.minor}' == expected_python, 'interpreter differs from configured existing environment')
    need(os.environ.get('SLURM_JOB_PARTITION') == 'standard', 'standard CPU allocation required')
    need(not any(os.environ.get(k) not in (None, '', '0') for k in ('SLURM_JOB_GPUS', 'SLURM_STEP_GPUS', 'SLURM_GPUS_ON_NODE', 'SLURM_GPUS')), 'GPU allocation prohibited')
    need(int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) == c['resources']['cpus'], 'CPU request differs from plan')
    memory = c['resources']['memory']
    expected_mb = int(memory[:-1]) * (1024 if memory[-1] == 'G' else 1)
    need(int(os.environ.get('SLURM_MEM_PER_NODE', '0')) == expected_mb, 'memory request differs from plan')
    job_info = subprocess.check_output(['scontrol', 'show', 'job', os.environ['SLURM_JOB_ID'], '-o'], text=True)
    limit = re.search(r'\bTimeLimit=(\S+)', job_info)
    def seconds(value):
        days, clock = value.split('-') if '-' in value else ('0', value)
        parts = list(map(int, clock.split(':')))
        if len(parts) == 2:
            parts.insert(0, 0)
        return int(days)*86400 + parts[0]*3600 + parts[1]*60 + parts[2]
    need(limit and seconds(limit[1]) == seconds(c['resources']['time']), 'walltime differs from plan')
    need(os.environ.get('SLURM_JOB_ACCOUNT') == c['account'], 'account differs from plan')
    need(ROOT == Path(c['remote_source']).resolve(), 'execution source differs from configured checkout')
    in_checkout = c.get('source_mode') == 'checkout'
    source = checkout_source(c) if in_checkout else verify_snapshot(ROOT / 'snapshot-manifest.json', expected)
    need(source['config_sha256'] == canonical(c), 'configuration differs from recorded source fingerprint')
    a = approval_path if in_checkout else load(approval_path)
    need(a['config_sha256'] == canonical(c) and (in_checkout or a['snapshot_manifest_sha256'] == expected) and a['resources_approved'] is True,
         'approval does not match configuration/source/resources')
    need(a.get('approved_by') and a.get('approved_utc'), 'human approval provenance required')
    if c['phase'] in ('development', 'frozen', 'ablations', 'sensitivity'):
        need(a.get('scientific_decisions_approved') is True, 'scientific decisions require approval')
        need(c['calibration']['outcome'] == 'passed' or a.get('provisional_approved') is True, 'provisional authorization required')
    if c['calibration']['outcome'] == 'passed':
        report = c['calibration'].get('report')
        need(report in source['files'] and source['files'][report] == c['calibration'].get('sha256'), 'matched calibration report missing from recorded source')
        evidence = load(ROOT / report)
        validate_calibration_evidence(evidence)
    prerequisites = {'checks': [], 'calibration': ['checks'], 'functional-smoke': ['checks', 'calibration'],
                     'ppo-update-smoke': ['functional-smoke'], 'development': ['ppo-update-smoke'],
                     'frozen': ['development'], 'ablations': ['frozen'], 'sensitivity': ['frozen'], 'final-checks': ['frozen']}
    observed = []
    for item in a.get('prerequisite_results', []):
        m = validate_completed(item['path'], item['sacct'])
        need(m['config']['agents'] == c['agents'], 'prerequisite agent inventory changed')
        need(m['source_code_sha256'] == canonical({k:v for k,v in source['files'].items() if k != 'run-config.json'}), 'prerequisite source changed; rerun checks')
        if c['phase'] == 'ablations' and m['phase'] == 'frozen':
            need(all(m['config'][k] == c[k] for k in ('n_train', 'n_test', 'n_validation', 'seeds', 'epochs')), 'ablation budget differs from frozen control')
        observed.append(m['phase'])
    if not in_checkout:
        need(set(prerequisites[c['phase']]) <= set(observed), 'missing validated prerequisite phase')
    if c['phase'] == 'functional-smoke' and c['calibration']['outcome'] != 'passed':
        need(a.get('provisional_approved') is True, 'incomplete calibration requires provisional authorization')
    out = Path(c['remote_results']) / c['run_id'] / 'artifacts'
    out.mkdir(parents=True, exist_ok=False)
    save(out/'source_manifest.json',source)
    if in_checkout:
        save(out/'config.json',c)
        patch=subprocess.check_output(['git','-C',str(ROOT),'diff','--binary','HEAD','--',*source['files']])
        (out/'source_changes.patch').write_bytes(patch)
        if c['phase']=='frozen': archive_frozen_source(source,c,out)
    m = {'run_id': c['run_id'], 'phase': c['phase'], 'config': c, 'config_sha256': canonical(c),
         'started_utc': datetime.now(timezone.utc).isoformat(), 'head_sha': source['head_sha'],
         'dirty_diff_sha256': source['dirty_diff_sha256'], 'snapshot_manifest_sha256': expected,
         'source_code_sha256': canonical({k:v for k,v in source['files'].items() if k != 'run-config.json'}),
         'source_manifest': source, 'approval': a, 'hostname': socket.gethostname(),
         'python': sys.version, 'python_executable': sys.executable, 'python_binary_sha256': digest(Path(sys.executable).resolve()), 'slurm_job_record': job_info, 'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
         'policy_training_device': 'cpu', 'rl_training_gpu_allocated': False,
         'slurm': {k:v for k,v in os.environ.items() if k.startswith('SLURM_')},
         'threads': {k:os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
         'label': 'FROZEN_SIMULATED' if c['phase'] == 'frozen' and c['calibration']['outcome'] == 'passed' else 'PROVISIONAL_UNCALIBRATED',
         'status': 'running', 'exit_code': None, 'commands': [], 'entry_argv': sys.argv.copy()}
    save(out / 'manifest.json', m)
    print(json.dumps({'run_id': c['run_id'], 'phase': c['phase'], 'host': m['hostname'],
                      'job_id': os.environ['SLURM_JOB_ID'], 'cpus': c['resources']['cpus'],
                      'device': 'cpu', 'snapshot_sha256': expected}), flush=True)
    def command(args, accepted=(0,)):
        m['commands'].append(args)
        with (out / 'commands.log').open('a') as log:
            result = subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        m.setdefault('command_exit_codes', []).append(result.returncode)
        if result.returncode not in accepted:
            raise subprocess.CalledProcessError(result.returncode, args)
        return result.returncode
    try:
        sys.path.insert(0, str(ROOT))
        m['dependency_check'] = dependency_report(ROOT / 'requirements.txt', m['packages'])
        save(out / 'dependencies.json', m['dependency_check'])
        need(all(r['matches'] for r in m['dependency_check']), 'installed packages differ from source pins; review dependencies.json, do not update the shared environment automatically')
        m['agents'] = inventory(c['agents'])
        from infrarl.simulator.environment import REWARD_WEIGHTS, max_reward
        if 'ppo' in c['agents'] or c['phase'] == 'ppo-update-smoke':
            import torch
            from infrarl.agents.ppo_agent import PPOAgent
            model = PPOAgent(seed=0)
            need(all(p.device.type == 'cpu' for p in model.net.parameters()), 'policy device not CPU')
            need(torch.as_tensor([0.0]).device.type == 'cpu', 'tensor device not CPU')
            m['ppo_hyperparameters'] = str(inspect.signature(PPOAgent))
            m['cuda_available_reporting_only'] = torch.cuda.is_available()
        weights = tuple(c.get('scenario', {}).get('reward_weights', REWARD_WEIGHTS))
        m['reward'] = {'weights': weights, 'quality_bonus': 0.3,
                       'maximum': sum(w * cap for w, cap in zip(weights, (1.3, 1.1, 1.0))),
                       'canonical_weights': REWARD_WEIGHTS, 'canonical_maximum': max_reward()}
        if c['phase'] in ('checks', 'final-checks'):
            command([sys.executable, '-m', 'pytest', '-q', 'tests', '-p', 'no:cacheprovider'])
            command([sys.executable, '-m', 'infrarl.evaluation.verify', '--out', str(out / 'verification.json')])
            command([sys.executable, 'scripts/generate_profile_exemplars.py', '--out', str(out / 'exemplars')])
        elif c['phase'] == 'calibration':
            command([sys.executable, 'scripts/run_calibration.py', '--out', str(out / 'calibration')], accepted=(0,1,2))
            report = load(out / 'calibration/summary.json')
            save(out / 'calibration.json', report)
            m['calibration_assessment'] = report['outcome']
            # Execution success means the honest report exists, NOT that V10 passed.
        elif c['phase'] == 'ppo-update-smoke':
            ppo_smoke(out)
        else:
            experiment(c, out)
        if in_checkout: verify_checkout(c,source)
        else: verify_snapshot(ROOT / 'snapshot-manifest.json', expected)
        current_packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
        need(current_packages == m['packages'], 'shared environment changed during job')
        m.update(status='execution-complete', exit_code=0)
        m['artifact_sha256'] = file_hashes(out)
        save(out / 'manifest.json', m)
        validate_artifacts(out)
    except BaseException as exc:
        m.update(status='failed', exit_code=getattr(exc, 'returncode', 1), error=str(exc))
        raise
    finally:
        m['ended_utc'] = datetime.now(timezone.utc).isoformat()
        m['artifact_sha256'] = file_hashes(out)
        save(out / 'manifest.json', m)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    for name in ('plan', 'snapshot', 'execute'):
        q = sub.add_parser(name)
        q.add_argument('--config', required=True)
        if name == 'snapshot':
            q.add_argument('--destination', required=True)
        if name == 'execute':
            q.add_argument('--approval', required=True)
            q.add_argument('--expected-sha256', required=True)
            q.add_argument('--phase', choices=PHASES, required=True)
    q = sub.add_parser('checkout', help='run a phase in the existing checkout, inside Slurm only')
    q.add_argument('phase', choices=PHASES)
    q.add_argument('--provisional', action='store_true')
    q.add_argument('--hyperparameters-frozen', action='store_true')
    q.add_argument('--calibration-report')
    q.add_argument('--agents')
    q.add_argument('--epochs', type=int)
    q.add_argument('--seeds', type=int)
    q.add_argument('--ablate', default='ppo')
    q = sub.add_parser('verify-snapshot')
    q.add_argument('--manifest', required=True)
    q.add_argument('--expected-sha256', required=True)
    q = sub.add_parser('validate')
    q.add_argument('--out', required=True)
    q.add_argument('--sacct', required=True, help='sacct --parsable2 output with JobID,State,ExitCode,Elapsed,MaxRSS')
    args = p.parse_args()
    if args.action == 'checkout':
        checkout_phase(args)
    elif args.action == 'verify-snapshot':
        verify_snapshot(args.manifest, args.expected_sha256)
        print('Snapshot verified')
    elif args.action == 'validate':
        validate_completed(args.out, args.sacct)
        print('Completed job and artifacts verified')
    else:
        c = config(args.config)
        if args.action == 'plan':
            plan(c)
        elif args.action == 'snapshot':
            snapshot(c, args.destination)
        else:
            need(args.phase == c['phase'], 'phase differs from configuration')
            execute(c, args.approval, args.expected_sha256)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError) as exc:
        raise SystemExit(str(exc))
