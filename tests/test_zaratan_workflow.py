"""Stdlib acceptance checks; also discovered by pytest. No SSH/Slurm calls."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import csv
import subprocess
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('zaratan_workflow', ROOT / 'scripts/zaratan/workflow.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


def example():
    return dict(run_id='test-run', phase='functional-smoke', agents=['random', 'heuristic', 'ppo'],
                account='reviewed_account', remote_source='/approved/source/test-run',
                remote_results='/approved/results', python='/approved/env/bin/python',
                resources=dict(partition='standard', cpus=4, memory='8G', time='00:15:00', gpus=0),
                n_train=50, n_validation=50, n_test=50, seeds=1, epochs=1,
                calibration=dict(outcome='unverified'))


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        cache = Path(os.environ.get('INFRARL_RUNTIME_CACHE', ROOT / '.workflow-validation'))
        cache.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=cache)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_config(self, c):
        path = self.root / 'config.json'
        w.save(path, c)
        return path

    def test_config_rejects_gpu_missing_inventory_and_invalid_splits(self):
        c = example()
        w.config(self.write_config(c))
        for change in (dict(agents=[]), dict(seeds=1001), dict(scenario={'reward_weights': [1, 0, 0]})):
            bad = {**c, **change}
            with self.assertRaises(ValueError):
                w.config(self.write_config(bad))
        c['resources']['gpus'] = 1
        with self.assertRaises(ValueError):
            w.config(self.write_config(c))

    def test_frozen_configuration_requires_inventory_and_budget(self):
        c = example()
        c.update(phase='frozen', n_train=1000, n_test=1000, seeds=3, epochs=None, hyperparameters_frozen=True)
        w.config(self.write_config(c))
        c['seeds'] = 1
        with self.assertRaises(ValueError):
            w.config(self.write_config(c))

    def test_shell_syntax(self):
        for name in ('setup.sh', 'sync.sh', 'sync_safe.sh', 'checkout.sh', 'run_experiment.sbatch'):
            subprocess.run(['bash', '-n', str(ROOT / 'scripts/zaratan' / name)], check=True)

    def test_environment_changes_disabled(self):
        result = subprocess.run(['bash', str(ROOT / 'scripts/zaratan/setup.sh'), 'install'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('environment changes are disabled', result.stderr)

    def test_generated_controls_require_approval_and_have_valid_shell(self):
        import sys
        spec = importlib.util.spec_from_file_location('prepare_run_test', ROOT / 'scripts/zaratan/prepare_run.py')
        prepare = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'workflow': w}):
            spec.loader.exec_module(prepare)
        c = example()
        target = self.root / 'control'
        prepare.controls(c, 'trusted-hash', target, [])
        approval = w.load(target / 'approval.json')
        self.assertFalse(approval['resources_approved'])
        script = (target / 'submit.sh').read_text()
        self.assertIn('--account=reviewed_account', script)
        self.assertIn('Resource approval has not been recorded', script)
        self.assertIn('verify-snapshot', script)
        subprocess.run(['bash', '-n', str(target / 'submit.sh')], check=True)
        preview = subprocess.run(['bash', str(target / 'submit.sh'), '--dry-run'], capture_output=True, text=True, check=True)
        self.assertIn('sbatch --parsable', preview.stdout)
        self.assertFalse((self.root/'artifacts').exists())

    def test_compute_entry_guard(self):
        from infrarl import require_compute_allocation
        with patch.dict(os.environ, {}, clear=True), patch('socket.gethostname', return_value='login-1.zaratan.umd.edu'):
            with self.assertRaises(RuntimeError): require_compute_allocation()
        with patch.dict(os.environ, {}, clear=True), patch('socket.gethostname', return_value='wsl-local'):
            require_compute_allocation()

    def test_checkout_clean_tree_and_fast_forward_with_fake_git(self):
        # No real Git mutations, commits, fetches, network or SSH: fake git records arguments.
        binpath=self.root/'bin';binpath.mkdir()
        log=self.root/'git.log'
        fake=binpath/'git'
        fake.write_text("""#!/usr/bin/python3
import os,sys,json
with open(os.environ['FAKE_GIT_LOG'],'a') as h: h.write(json.dumps(sys.argv[1:])+'\\n')
a=sys.argv[3:]
if a[0]=='status': print(' M personal.py' if os.environ.get('FAKE_DIRTY') else '',end='')
elif a[0]=='branch': print('main')
elif a[0]=='rev-parse': print(os.environ['FAKE_STATE']+'/'+a[-1])
elif a[0]=='merge-base' and os.environ.get('FAKE_DIVERGED'): sys.exit(1)
""")
        fake.chmod(0o755)
        env={**os.environ,'PATH':str(binpath)+':'+os.environ['PATH'], 'FAKE_GIT_LOG':str(log),'FAKE_STATE':str(self.root/'state')}
        cmd=['bash',str(ROOT/'scripts/zaratan/checkout.sh'),'update',str(self.root)]
        result=subprocess.run(cmd+['--dry-run'],env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertNotIn('fetch',log.read_text())
        result=subprocess.run(cmd,env={**env,'FAKE_DIRTY':'yes'},capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertNotIn('fetch',log.read_text())
        result=subprocess.run(cmd,env={**env,'FAKE_DIVERGED':'yes'},capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertNotIn('--ff-only',log.read_text())
        result=subprocess.run(cmd,env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--ff-only',log.read_text())

    def test_control_transfer_preview_has_no_ssh_or_delete(self):
        control=self.root/'preview-control';control.mkdir()
        for name in ('approval.json','run-info.json','submit.sh'):
            (control/name).write_text('{}')
        binpath=self.root/'preview-bin';binpath.mkdir()
        log=self.root/'rsync-args.txt'
        (binpath/'rsync').write_text("#!/bin/bash\nprintf '%s\\n' \"$@\" > \"$TEST_RSYNC_LOG\"\n")
        (binpath/'ssh').write_text('#!/bin/bash\necho unexpected-ssh >&2\nexit 99\n')
        (binpath/'rsync').chmod(0o755);(binpath/'ssh').chmod(0o755)
        env={**os.environ,'PATH':str(binpath)+':'+os.environ['PATH'],'TEST_RSYNC_LOG':str(log)}
        result=subprocess.run(['bash',str(ROOT/'scripts/zaratan/sync_safe.sh'),'controls','--dry-run',str(control),'unit@login.zaratan.umd.edu:/approved/run'],env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--dry-run',log.read_text())
        self.assertNotIn('--delete',log.read_text())

    def test_checkout_config_keeps_outputs_in_existing_project(self):
        c=example();c.update(source_mode='checkout',remote_source='/approved/project',remote_results='/approved/project/results')
        self.assertEqual(w.validate_config(c)['source_mode'],'checkout')
        c.pop('source_mode')
        with self.assertRaises(ValueError): w.validate_config(c)

    def test_checkout_phase_requires_allocation_and_approval(self):
        with patch.dict(os.environ,{},clear=True), patch.object(w.subprocess,'check_output',side_effect=AssertionError('must not query Slurm')):
            with self.assertRaises(ValueError): w.checkout_phase(None)
        with patch.dict(os.environ,{'SLURM_JOB_ID':'42'},clear=True), patch.object(w.subprocess,'check_output',side_effect=AssertionError('must not query Slurm')):
            with self.assertRaises(ValueError): w.checkout_phase(None)

    def test_checkout_phase_reuses_executor_without_source_copy(self):
        import argparse
        args=argparse.Namespace(phase='calibration',agents=None,epochs=None,seeds=None,
                ablate='ppo',calibration_report=None,hyperparameters_frozen=False,provisional=False)
        env={'SLURM_JOB_ID':'42','SLURM_JOB_ACCOUNT':'reviewed_account','SLURM_CPUS_PER_TASK':'4',
             'SLURM_MEM_PER_NODE':'8192','INFRARL_RESOURCES_APPROVED':'yes','USER':'unit-review'}
        seen=[]
        def record(c,a,expected):
            w.validate_config(c);seen.append((c,a,expected))
        with patch.dict(os.environ,env,clear=True),patch.object(w.subprocess,'check_output',return_value='TimeLimit=02:00:00'),patch.object(w,'execute',side_effect=record):
            w.checkout_phase(args)
        c,a,expected=seen[0]
        self.assertEqual(c['remote_source'],str(ROOT))
        self.assertEqual(c['remote_results'],str(ROOT/'results'))
        self.assertEqual(c['source_mode'],'checkout')
        self.assertIsNone(expected)
        self.assertTrue(a['resources_approved'])

    def test_checkout_hashes_and_final_archive(self):
        import tarfile
        root,_,_,_=self.snapshot_fixture()
        def git(args):
            if args[-1]=='ls-files': return b'infrarl/tracked.py\n'
            if args[-2:]==['rev-parse','HEAD']: return b'abc123\n'
            return b'dirty diff'
        c=example()
        with patch.object(w,'ROOT',root),patch.object(w.subprocess,'check_output',side_effect=git):
            source=w.checkout_source(c)
            w.verify_checkout(c,source)
            self.assertEqual(source['included_untracked'],['infrarl/new.py'])
            archive_dir=self.root/'frozen';archive_dir.mkdir()
            w.archive_frozen_source(source,c,archive_dir)
            archive=archive_dir/'frozen_source.tar'
            self.assertEqual(archive.stat().st_mode & 0o222,0)
            with tarfile.open(archive) as h:
                self.assertEqual(h.extractfile('infrarl/new.py').read(),b'untracked code\n')
                self.assertEqual(json.load(h.extractfile('__run_config.json')),c)
            (root/'infrarl/tracked.py').write_text('changed while running')
            with self.assertRaises(ValueError): w.verify_checkout(c,source)

    def test_checkout_overlay_preview_has_backups_and_no_delete(self):
        root=self.root/'overlay';(root/'scripts/zaratan').mkdir(parents=True)
        (root/'module.py').write_text('selected local code')
        (root/'scripts/zaratan/transfer_files.txt').write_text('module.py\n')
        binpath=self.root/'overlay-bin';binpath.mkdir()
        log=self.root/'overlay-args'
        (binpath/'rsync').write_text("#!/bin/bash\nprintf '%s\\n' \"$@\" >> \"$TEST_RSYNC_LOG\"\n".replace('\\"','"'))
        (binpath/'ssh').write_text('#!/bin/bash\nexit 99\n')
        for name in ('ssh','rsync'): (binpath/name).chmod(0o755)
        env={**os.environ,'PATH':str(binpath)+':'+os.environ['PATH'],'TEST_RSYNC_LOG':str(log),'INFRARL_TRANSFER_BACKUP':'/approved/backups/review'}
        result=subprocess.run(['bash',str(ROOT/'scripts/zaratan/sync_safe.sh'),'checkout','--dry-run',str(root),'unit@login.zaratan.umd.edu:/approved/project'],env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--files-from=',log.read_text())
        self.assertIn('--backup-dir=/approved/backups/review',log.read_text())
        self.assertIn('--dry-run',log.read_text())
        self.assertNotIn('--delete',log.read_text())
        manifests=list((root/'.workflow-validation').glob('*.sha256'))
        self.assertEqual(len(manifests),1)
        self.assertIn('module.py',manifests[0].read_text())

    def test_plan_never_submits(self):
        with patch.object(w.subprocess, 'run', side_effect=AssertionError('must not execute')), patch('builtins.print') as output:
            w.plan(example())
        self.assertIn('sbatch', ' '.join(str(call) for call in output.call_args_list))

    def snapshot_fixture(self):
        root = self.root / 'repo'
        (root / 'infrarl').mkdir(parents=True)
        (root / 'infrarl/tracked.py').write_text('dirty tracked content\n')
        (root / 'infrarl/new.py').write_text('untracked code\n')
        (root / 'infrarl/.env.json').write_text('secret')
        (root / 'hpc').mkdir()
        (root / 'hpc/personal.sbatch').write_text('untouched')
        # Hidden content must fail packaging, not silently leak.
        with self.assertRaises(ValueError):
            w.source_files(root)
        (root / 'infrarl/.env.json').unlink()
        dest = self.root / 'snapshot'
        def git_output(args):
            if args[-1] == 'ls-files':
                return b'infrarl/tracked.py\n'
            if args[-2:] == ['rev-parse', 'HEAD']:
                return b'abc123\n'
            return b'dirty diff'
        with patch.object(w.subprocess, 'check_output', side_effect=git_output):
            w.snapshot(example(), dest, root=root)
        path = dest / 'snapshot-manifest.json'
        return root, dest, path, w.digest(path)

    def test_snapshot_includes_dirty_untracked_and_excludes_hpc(self):
        root, dest, path, sha = self.snapshot_fixture()
        m = w.verify_snapshot(path, sha)
        self.assertEqual(m['included_untracked'], ['infrarl/new.py'])
        self.assertNotIn('hpc/personal.sbatch', m['files'])
        self.assertEqual((dest / 'infrarl/tracked.py').read_text(), 'dirty tracked content\n')
        self.assertEqual((root / 'hpc/personal.sbatch').read_text(), 'untouched')
        self.assertEqual(w.digest(dest / 'infrarl/new.py'), m['files']['infrarl/new.py'])

    def test_snapshot_tampering_missing_and_unexpected_files_fail(self):
        _, dest, path, sha = self.snapshot_fixture()
        target = dest / 'infrarl/new.py'
        target.chmod(0o644)
        target.write_text('changed')
        with self.assertRaises(ValueError):
            w.verify_snapshot(path, sha)
        target.write_text('untracked code\n')
        extra = dest / 'unexpected.py'
        extra.write_text('x')
        with self.assertRaises(ValueError):
            w.verify_snapshot(path, sha)
        extra.unlink()
        target.unlink()
        with self.assertRaises(ValueError):
            w.verify_snapshot(path, sha)

    def test_execute_rejects_login_and_gpu_allocation(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            w.execute(example(), 'never-opened', 'none')
        with patch.dict(os.environ, {'SLURM_JOB_ID': '12', 'SLURM_JOB_PARTITION': 'gpu'}, clear=True), self.assertRaises(ValueError):
            w.execute(example(), 'never-opened', 'none')

    def completion_fixture(self):
        out = self.root / 'artifacts'
        out.mkdir()
        w.save(out / 'ppo-update.json', dict(optimizer_steps=1, finite=True, parameters_changed=True))
        c = example()
        c['phase'] = 'ppo-update-smoke'
        m = dict(config=c, config_sha256=w.canonical(c), status='execution-complete', exit_code=0,
                 slurm={'SLURM_JOB_ID': '42'}, artifact_sha256=w.file_hashes(out))
        w.save(out / 'manifest.json', m)
        (self.root / 'logs').mkdir()
        (self.root / 'logs/infrarl-42.out').write_text('test job output')
        accounting = self.root / 'sacct.txt'
        accounting.write_text('JobID|State|ExitCode|Elapsed|MaxRSS\n42|COMPLETED|0:0|00:01:00|\n42.batch|COMPLETED|0:0|00:01:00|10M\n')
        return out, accounting

    def test_completion_checks_accounting_and_hashes(self):
        out, accounting = self.completion_fixture()
        w.validate_completed(out, accounting)
        accounting.write_text('JobID|State|ExitCode\n42|TIMEOUT|0:0\n')
        with self.assertRaises(ValueError):
            w.validate_completed(out, accounting)
        accounting.write_text('JobID|State|ExitCode\n42|COMPLETED|0:0\n42.batch|FAILED|1:0\n')
        with self.assertRaises(ValueError):
            w.validate_completed(out, accounting)
        (out / 'ppo-update.json').write_text('{}')
        with self.assertRaises(ValueError):
            w.validate_artifacts(out)

    def test_existing_environment_pin_check_never_installs(self):
        path = self.root / 'requirements.txt'
        path.write_text('# pins\nSome_Package==1.0\nmissing==2.0\n')
        rows = w.dependency_report(path, {'some-package': '1.0'})
        self.assertTrue(rows[0]['matches'])
        self.assertFalse(rows[1]['matches'])
        self.assertIsNone(rows[1]['installed'])
        rows = w.dependency_report(path, {'some-package': '1.1', 'missing': '2.0'})
        self.assertFalse(rows[0]['matches'])

    def test_inventory_fails_if_named_agent_missing(self):
        registry = types.ModuleType('infrarl.agents.registry')
        registry.REGISTRY = {'random': types.SimpleNamespace(train_epochs=0, factory=lambda seed: object())}
        with patch.dict('sys.modules', {'infrarl.agents.registry': registry}):
            self.assertEqual(w.inventory(['random'])['random']['train_epochs'], 0)
            with self.assertRaises(ValueError):
                w.inventory(['random', 'dqn'])

    def test_calibration_requires_matched_complete_references(self):
        with self.assertRaises(ValueError):
            w.validate_calibration_evidence(dict(complete=True, outcome='passed', matched_references=True))
        cell = dict(id='synthetic-test-cell', outcome='passed', comparable=True, independent_reference=True,
                    reference=100, prediction=115, source_url='https://example.invalid/test-fixture',
                    retrieved_utc='test', model_id='test', hardware='test', precision='test',
                    workload='test', metric='test', units='test')
        evidence = dict(complete=True, outcome='passed', matched_references=True,
                        required_cells=[cell['id']], cells=[cell])
        w.validate_calibration_evidence(evidence)
        cell['prediction'] = 115.01
        with self.assertRaises(ValueError):
            w.validate_calibration_evidence(evidence)
        cell['reference'] = 0
        with self.assertRaises(ValueError):
            w.validate_calibration_evidence(evidence)

    def test_episode_counts_metrics_and_nonempty_files(self):
        out = self.root / 'episode-artifacts'
        e = out / 'experiment'
        (e / 'figures').mkdir(parents=True)
        c = example()
        c.update(phase='development', agents=['random'], n_validation=2)
        metrics = dict(avg_quality=.8, quality_attainment=1, quality_compliance=1,
                       budget_compliance=1, sla_compliance=1, meets_all=1, mean_reward=1, mean_cost_usd=1)
        def write(name, rows, fields=None):
            with (e / name).open('w') as h:
                writer = csv.DictWriter(h, fieldnames=fields or list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        episodes = [dict(agent='random', seed=0, profile_idx=i, quality=.8, latency_ms=1,
                         cost_usd=1, reward=1, acc_req=.7, budget=2, sla_ms=2,
                         within_budget=1, meets_sla=1, meets_quality=1) for i in range(2)]
        write('episodes.csv', episodes)
        write('summary_per_seed.csv', [dict(agent='random', seed=0, **metrics)])
        write('summary.csv', [dict(agent='random', mean_reward_mean=1)])
        write('curves.csv', [], ['agent', 'seed', 'epoch', 'train_reward', 'eval_reward'])
        write('stats.csv', [], ['seed', 'metric', 'test'])
        for name in ('pareto_quality_vs_cost.png', 'heatmap_meets_all_by_use_case.png', 'compliance.png'):
            (e / 'figures' / name).write_bytes(b'nonempty-test-fixture')
        w.save(e / 'experiment.json', dict(agents='random', curve_split='validation', validation_seed=3000,
               test_seed=1000, evaluation_split='validation', budgets={'random': {'epochs': 0}}))
        m = dict(config=c, config_sha256=w.canonical(c), status='execution-complete', exit_code=0,
                 reward={'maximum': 1.18}, artifact_sha256=w.file_hashes(out))
        def refresh():
            m['artifact_sha256'] = w.file_hashes(out)
            w.save(out / 'manifest.json', m)
        refresh()
        w.validate_artifacts(out)
        write('episodes.csv', episodes[:1])
        refresh()
        with self.assertRaisesRegex(ValueError, 'count/coverage'):
            w.validate_artifacts(out)
        episodes[0]['reward'] = float('nan')
        write('episodes.csv', episodes)
        refresh()
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            w.validate_artifacts(out)

    def test_final_test_never_used_for_curves(self):
        # Execute the real run_one body with recording stubs, without numerical dependencies.
        tree = ast.parse((ROOT / 'infrarl/evaluation/run_experiment.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_one')
        calls = []
        namespace = dict(train=lambda *a, **kw: calls.append(('curve', kw['eval_profiles'])) or [],
                         evaluate=lambda e, a, p: calls.append(('final', p)) or [])
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<run_one>', 'exec'), namespace)
        class Spec:
            train_epochs = 2
            @staticmethod
            def factory(seed):
                return object()
        namespace['run_one'](Spec(), 0, ['train'], ['TEST'], lambda *a: None, validation_profiles=['VALIDATION'])
        self.assertEqual(calls, [('curve', ['VALIDATION']), ('final', ['TEST'])])
        calls.clear()
        namespace['run_one'](Spec(), 0, ['train'], ['TEST'], lambda *a: None)
        self.assertEqual(calls[0], ('curve', None))


if __name__ == '__main__':
    unittest.main()
