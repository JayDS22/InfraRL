#!/usr/bin/env python3
"""CPU residual arithmetic; use the calibration Slurm phase on Zaratan."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from infrarl.simulator.calibration import run


def main():
    from infrarl import require_compute_allocation
    require_compute_allocation()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--references', default=str(ROOT/'data/benchmarks/calibration_references.csv'))
    p.add_argument('--out', required=True)
    args = p.parse_args()
    rows, summary = run(args.references, args.out)
    print(summary['outcome'])
    return 0 if summary['complete'] else 1 if summary['outcome'] in ('failed', 'infeasible') else 2


if __name__ == '__main__':
    raise SystemExit(main())
