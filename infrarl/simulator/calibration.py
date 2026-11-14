"""Source-matched residual assessment; never fits or mutates simulator tables.

Exit semantics belong to the CLI: 0 complete pass, 1 confirmed failure/infeasibility,
2 incomplete evidence. Original infeasible anchors are retained as nonrequired tests.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from decimal import Decimal

from .hardware import HARDWARE
from .models import min_memory_gb

STATUS_OUTCOME = {'PASS': 'passed', 'FAIL': 'failed', 'INFEASIBLE': 'infeasible',
                  'MISSING_REFERENCE': 'unverified', 'NOT_COMPARABLE': 'incomparable'}
AXIS_UNITS = {'quality': 'fraction', 'latency': 'ms', 'cost': 'USD/month'}


def residual(prediction, reference, comparable=True, tolerance=15):
    """Return status/error/reason; absent, nonfinite and zero references never divide."""
    if reference is None or reference == '':
        return 'MISSING_REFERENCE', None, 'No independently verified reference'
    try:
        ref = Decimal(str(reference))
        pred = Decimal(str(prediction))
    except Exception:
        return 'NOT_COMPARABLE', None, 'Missing or nonnumeric prediction/reference'
    if not ref.is_finite() or not pred.is_finite() or ref <= 0:
        return 'NOT_COMPARABLE', None, 'Reference must be positive; all values finite'
    if not comparable:
        return 'NOT_COMPARABLE', None, 'Model/hardware/workload/measurement definitions do not match'
    error = 100 * abs(pred-ref) / abs(ref)
    return ('PASS' if error <= Decimal(str(tolerance)) else 'FAIL'), float(error), 'Matched-axis relative error; tolerance <=15%'


def comparable(row):
    """Require reviewed provenance and explicit mapping from size bucket to checkpoint."""
    required = ('source_url', 'retrieved_utc', 'model_id', 'sim_model_id', 'provider',
                'hardware_id', 'sim_hardware_id', 'gpu_count', 'precision', 'runtime',
                'workload', 'statistic', 'publication_date', 'measurement_type', 'sim_measurement_type')
    if row.get('hardware') != 'cpu':
        try:
            if float(row.get('gpu_vram_gb', '')) != HARDWARE[row['hardware']]['memory_gb'] or int(row.get('gpu_count', '')) != 1:
                return False
        except (ValueError, KeyError):
            return False
    return (row.get('reference_status') == 'MATCHED_REFERENCE'
            and all(row.get(k) for k in required)
            and row['model_id'] == row['sim_model_id']
            and row['hardware_id'] == row['sim_hardware_id']
            and row['measurement_type'] == row['sim_measurement_type']
            and row.get('independent_reference') == 'true'
            and row.get('units') == AXIS_UNITS.get(row.get('axis')))


def prediction_for(row):
    # Import numerical simulator only when a comparable reference/workload exists.
    from .estimator import Tier, Stack, is_feasible, estimate
    from .profiles import Profile
    tier = Tier(row['hardware'], row['model'], row['precision'])
    if not is_feasible(tier):
        raise ValueError('infeasible tier')
    numeric = ('monthly_queries', 'latency_sla_ms', 'acc_req', 'monthly_budget_usd',
               'peak_multiplier', 'avg_input_tokens', 'avg_output_tokens', 'concurrent_users')
    values = {k: float(row[k]) for k in numeric}
    if not all(math.isfinite(v) and v > 0 for v in values.values()):
        raise ValueError('finite positive workload fields required')
    profile = Profile(use_case=row['use_case'], data_sensitivity=row['data_sensitivity'], **values)
    q, latency, cost = estimate(Stack((tier,), row.get('scaling') or 'static'), profile)
    return {'quality': q, 'latency': latency, 'cost': cost}[row['axis']]


def assess(row, predictor=prediction_for):
    result = {k: row.get(k, '') for k in ('id', 'config_id', 'hardware', 'model', 'precision', 'axis',
               'reference', 'units', 'source_url', 'role', 'required', 'model_id', 'hardware_id',
               'runtime', 'workload', 'retrieved_utc', 'independent_reference', 'reference_status')}
    result.update(prediction='', residual_pct='', status='', outcome='', reason='')
    try:
        fits = min_memory_gb(row['model'], row['precision']) <= HARDWARE[row['hardware']]['memory_gb']
    except KeyError as exc:
        raise ValueError('Unknown calibration hardware/model/precision') from exc
    if not fits:
        status, error, reason = 'INFEASIBLE', None, 'Weights-only memory exceeds one device; never estimated or fitted'
    elif row.get('reference', '') == '':
        status, error, reason = residual(None, None)
    elif not comparable(row):
        status, error, reason = 'NOT_COMPARABLE', None, 'Incomplete provenance, checkpoint mapping or incompatible metric/configuration'
    else:
        try:
            if row['axis'] == 'quality' and not (0 < float(row['reference']) <= 1):
                raise ValueError('quality reference must be a fraction in (0,1]')
            result['prediction'] = predictor(row)
            status, error, reason = residual(result['prediction'], row['reference'])
        except (KeyError, ValueError) as exc:
            status, error, reason = 'NOT_COMPARABLE', None, 'Workload/configuration invalid: ' + str(exc)
    result.update(status=status, outcome=STATUS_OUTCOME[status], residual_pct='' if error is None else error, reason=reason)
    return result


def summarize(rows):
    required = [r for r in rows if r['required'] == 'true' and r['role'] == 'calibration']
    holdout = [r for r in rows if r['role'] == 'holdout']
    complete = bool(required) and all(r['status'] == 'PASS' for r in required)
    statuses = {r['status'] for r in required}
    outcome = ('passed' if complete else 'failed' if 'FAIL' in statuses else
               'infeasible' if 'INFEASIBLE' in statuses else
               'incomparable' if 'NOT_COMPARABLE' in statuses else 'unverified')
    return {'outcome': outcome, 'complete': complete, 'matched_references': complete,
            'required_cells': [r['id'] for r in required],
            'counts': {s: sum(r['status'] == s for r in rows) for s in STATUS_OUTCOME},
            'holdout_cells': len(holdout), 'holdout_passed': bool(holdout) and all(r['status'] == 'PASS' for r in holdout),
            'cells': [{**r, 'hardware': r['hardware_id'], 'comparable': r['status'] in ('PASS', 'FAIL'),
                       'independent_reference': r['independent_reference'] == 'true', 'metric': r['axis']} for r in required]}


def render_report(rows, summary):
    text = ['# Calibration residual assessment', '',
            f"Gate: **{summary['outcome'].upper()}**; complete required coverage: {summary['complete']}.",
            'Tolerance: <=15% only for independently matched comparable axes. Original infeasible tests are excluded from required coverage.', '',
            '| Configuration | Axis | Prediction | Reference | Units | Status | Reason |',
            '|---|---|---|---|---|---|---|']
    for r in rows:
        text.append('| ' + ' | '.join(str(r[k]).replace('|', '/') for k in ('config_id', 'axis', 'prediction', 'reference', 'units', 'status', 'reason')) + ' |')
    text += ['', 'Unmatched required axes: ' + ', '.join(r['id'] for r in rows if r['required'] == 'true' and r['status'] != 'PASS'),
             '', f"Independent holdout cells: {summary['holdout_cells']}. A complete anchor pass alone does not establish generalization.", '']
    return '\n'.join(text)


def run(references, out):
    references, out = Path(references), Path(out)
    with references.open() as handle:
        records = list(csv.DictReader(handle))
    ids = [r['id'] for r in records]
    keys = [(r['config_id'], r['axis'], r['role']) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate configuration/axis/role reference rows')
    if not records or len(ids) != len(set(ids)):
        raise ValueError('Reference rows must have unique IDs')
    # The intended three primary anchors must each declare the full 3-axis matrix.
    primary = ('cpu_3b_int8', 'a10_13b_int8', 'a100_70b_int8')
    declared = {(r['config_id'], r['axis']) for r in records if r['required'] == 'true' and r['role'] == 'calibration'}
    if not {(c, a) for c in primary for a in AXIS_UNITS} <= declared:
        raise ValueError('Required revised-anchor 3x3 matrix is incomplete')
    rows = [assess(r) for r in records]
    summary = summarize(rows)
    summary['references_sha256'] = hashlib.sha256(references.read_bytes()).hexdigest()
    summary['source_revision'] = 'See immutable source snapshot/run manifest; no parameter fitting performed'
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'residuals.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (out / 'report.md').write_text(render_report(rows, summary))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
    return rows, summary
