#!/usr/bin/env python3
"""Select six real generated rows, never hand-authored profiles or recommendations."""
from dataclasses import asdict
from pathlib import Path
import argparse
import csv
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def select_exemplars(profiles):
    import numpy as np
    from infrarl.simulator.models import USE_CASES
    numeric = ('monthly_queries', 'latency_sla_ms', 'acc_req', 'monthly_budget_usd',
               'peak_multiplier', 'avg_input_tokens', 'avg_output_tokens', 'concurrent_users')
    logarithmic = {'monthly_queries', 'latency_sla_ms', 'monthly_budget_usd',
                   'avg_input_tokens', 'avg_output_tokens', 'concurrent_users'}
    selected = []
    for use_case in USE_CASES:
        candidates = [(i,p) for i,p in enumerate(profiles) if p.use_case == use_case]
        if not candidates:
            raise ValueError('Missing use case: '+use_case)
        features = np.asarray([[np.log(getattr(p,k)) if k in logarithmic else getattr(p,k)
                                for k in numeric] for _,p in candidates])
        median = np.median(features, axis=0)
        iqr = np.percentile(features,75,axis=0)-np.percentile(features,25,axis=0)
        iqr = np.where(iqr > 0, iqr, 1)
        distances = np.sum(((features-median)/iqr)**2,axis=1)
        # np.argmin returns the first candidate; candidates preserve original index.
        index, profile = candidates[int(np.argmin(distances))]
        selected.append({'profile_index': index, **asdict(profile)})
    return selected


def generate(out):
    from infrarl.simulator.profiles import generate_profiles
    rows = select_exemplars(generate_profiles(1000, seed=1000))
    out = Path(out)
    out.mkdir(parents=True,exist_ok=False)
    with (out/'profile_exemplars.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    columns=list(rows[0])
    text=['# Six profile exemplars', '', '**Generated synthetic examples**, not customer records.',
          'Dataset: generate_profiles(1000, seed=1000). One actual row per use case.',
          'Selection: nearest featurewise category median by squared distance after log-transforming queries, SLA, budget, token lengths and concurrency, then median/IQR scaling. Zero IQR uses scale 1. Accuracy and peak multiplier remain linear; sensitivity is categorical and excluded from distance. Ties use the original profile index.', '',
          '| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for row in rows:
        text.append('| '+' | '.join(str(row[k]) for k in columns)+' |')
    (out/'PROFILE_EXEMPLARS.md').write_text('\n'.join(text)+'\n')
    return rows


if __name__=='__main__':
    from infrarl import require_compute_allocation
    require_compute_allocation()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,help='fresh output directory; never overwrites')
    generate(p.parse_args().out)
