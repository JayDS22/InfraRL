"""Small residual arithmetic tests run without training or external dependencies."""
import copy
import csv
import importlib.util
import tempfile
import os
import unittest
from pathlib import Path

from infrarl.simulator.calibration import assess, comparable, residual, run
from infrarl.simulator.hardware import HARDWARE, HOURS_PER_MONTH
from infrarl.simulator.models import BASE_QUALITY

ROOT=Path(__file__).resolve().parents[1]


def matched_row():
    row=dict(id='fixture',config_id='cpu_3b_int8',hardware='cpu',model='3b',precision='int8',
             axis='quality',reference='.8',units='fraction',required='true',role='calibration',
             reference_status='MATCHED_REFERENCE',independent_reference='true',model_id='test-model',
             sim_model_id='test-model',hardware_id='test-cpu',sim_hardware_id='test-cpu',gpu_count='1',
             source_url='https://example.invalid/unit-test',retrieved_utc='unit-test',publication_date='unit-test',
             provider='test',runtime='test',workload='test',statistic='test',measurement_type='test',sim_measurement_type='test')
    return row


class CalibrationTests(unittest.TestCase):
    def test_boundary_and_failure(self):
        self.assertEqual(residual(115,100)[0],'PASS')
        self.assertEqual(residual(115.001,100)[0],'FAIL')
        self.assertEqual(residual(.115,.1)[0],'PASS')

    def test_missing_zero_nonfinite_mismatch(self):
        self.assertEqual(residual(1,None)[0],'MISSING_REFERENCE')
        for ref in (0,-1,float('nan'),float('inf')):
            self.assertEqual(residual(1,ref)[0],'NOT_COMPARABLE')
        self.assertEqual(residual(1,1,False)[0],'NOT_COMPARABLE')

    def test_physical_invalidity_before_prediction(self):
        for hw,model in (('a10','13b'),('a100','70b')):
            row=matched_row();row.update(hardware=hw,model=model,precision='fp16')
            self.assertEqual(assess(row,predictor=lambda r: self.fail('must not estimate infeasible tier'))['status'],'INFEASIBLE')

    def test_hardware_memory_classes_cannot_be_relabelled(self):
        row=matched_row();row.update(hardware='a100',gpu_count='1',gpu_vram_gb='40')
        self.assertFalse(comparable(row))
        row['gpu_vram_gb']='80'
        self.assertTrue(comparable(row))

    def test_unmatched_and_mutation(self):
        tables=copy.deepcopy((HARDWARE,BASE_QUALITY))
        row=matched_row()
        self.assertEqual(assess(row,predictor=lambda r:.92)['status'],'PASS')
        row['sim_model_id']='different'
        self.assertFalse(comparable(row))
        self.assertEqual(assess(row,predictor=lambda r:self.fail('not comparable'))['status'],'NOT_COMPARABLE')
        self.assertEqual((HARDWARE,BASE_QUALITY),tables)
        self.assertEqual(HOURS_PER_MONTH,720)

    def test_seed_reference_report_is_incomplete_and_deterministic(self):
        cache=Path(os.environ.get('INFRARL_RUNTIME_CACHE',ROOT/'.workflow-validation'));cache.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temp:
            p=Path(temp)
            rows,summary=run(ROOT/'data/benchmarks/calibration_references.csv',p/'first')
            run(ROOT/'data/benchmarks/calibration_references.csv',p/'second')
            self.assertFalse(summary['complete']);self.assertEqual(summary['outcome'],'unverified')
            self.assertEqual(summary['counts']['INFEASIBLE'],6)
            self.assertEqual(summary['counts']['MISSING_REFERENCE'],9)
            for name in ('residuals.csv','report.md','summary.json'):
                self.assertEqual((p/'first'/name).read_bytes(),(p/'second'/name).read_bytes())

    def test_declared_matrix_cannot_silently_drop_axis(self):
        cache=Path(os.environ.get('INFRARL_RUNTIME_CACHE',ROOT/'.workflow-validation'));cache.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temp:
            p=Path(temp)
            with (ROOT/'data/benchmarks/calibration_references.csv').open() as f:
                reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
            with (p/'references.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows[1:])
            with self.assertRaises(ValueError):run(p/'references.csv',p/'out')


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'existing numerical environment unavailable')
class ExemplarTests(unittest.TestCase):
    def test_actual_rows_six_categories_determinism_and_no_overwrite(self):
        spec=importlib.util.spec_from_file_location('exemplars',ROOT/'scripts/generate_profile_exemplars.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        from infrarl.simulator.profiles import generate_profiles
        from infrarl.simulator.models import USE_CASES
        from dataclasses import asdict
        profiles=generate_profiles(1000,1000)
        rows=module.select_exemplars(profiles)
        self.assertEqual([r['use_case'] for r in rows],list(USE_CASES))
        for row in rows:
            self.assertEqual({k:v for k,v in row.items() if k!='profile_index'},asdict(profiles[row['profile_index']]))
            self.assertTrue(.6<=row['acc_req']<=.9)
            self.assertTrue(all(row[k]>0 for k in ('monthly_queries','latency_sla_ms','monthly_budget_usd','avg_input_tokens','avg_output_tokens','concurrent_users')))
        cache=Path(os.environ.get('INFRARL_RUNTIME_CACHE',ROOT/'.workflow-validation'));cache.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temp:
            p=Path(temp);module.generate(p/'a');module.generate(p/'b')
            for name in ('profile_exemplars.csv','PROFILE_EXEMPLARS.md'):
                self.assertEqual((p/'a'/name).read_bytes(),(p/'b'/name).read_bytes())
            with self.assertRaises(FileExistsError):module.generate(p/'a')


class VerificationProtocolTests(unittest.TestCase):
    def test_cli_oracle_uses_validation_not_final_test_seed(self):
        # Exercise actual CLI code with expensive numerical checks replaced by spies.
        import ast,argparse,json,sys,io,contextlib
        from unittest.mock import patch
        tree=ast.parse((ROOT/'infrarl/evaluation/verify.py').read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        oracle=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='oracle_gap')
        self.assertEqual(ast.literal_eval(oracle.args.defaults[-1]),3000)
        seen=[]
        namespace={'argparse':argparse,'json':json,'oracle_gap':lambda seed: seen.append(seed) or {'profile_seed':seed},
                   'check_price_anchor':lambda:{}}
        for name in ('check_masks','check_simulator','check_reward','check_determinism'):
            namespace[name]=lambda:{'n_failed':0}
        exec(compile(ast.Module(body=[main],type_ignores=[]),'<verification-cli>','exec'),namespace)
        cache=Path(os.environ.get('INFRARL_RUNTIME_CACHE',ROOT/'.workflow-validation'))
        cache.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temp:
            path=Path(temp)/'verification.json'
            with patch.object(sys,'argv',['verify','--out',str(path)]),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as result: namespace['main']()
            self.assertEqual(result.exception.code,0)
            self.assertEqual(seen,[3000])
            self.assertEqual(json.loads(path.read_text())['oracle']['profile_seed'],3000)
