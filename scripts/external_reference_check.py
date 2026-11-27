"""Compare simulator constants against sourced public data.

    python scripts/external_reference_check.py [--refs data/benchmarks/external_references_20261127.csv]

Checks the simulator's primitive constants (not end-to-end episodes), one row per
reference: GPU/CPU hourly price, single-stream decode tokens/s, per-size quality,
and the quantization quality penalty. Predictions are read from the live simulator
tables, so the check stays correct if the tables change. PASS = within 15% (the
interim report's gate); for quantization drop the gate is +/-0.01 absolute because
the values are tiny. Writes a markdown table to stdout.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from infrarl.simulator.estimator import Tier, tokens_per_sec  # noqa: E402
from infrarl.simulator.hardware import HARDWARE  # noqa: E402
from infrarl.simulator.models import BASE_QUALITY, PRECISION_QUALITY_PENALTY  # noqa: E402

TOL_PCT = 15.0
TOL_ABS = 0.01


def predict(axis: str, key: str) -> float:
    if axis == "price":
        return HARDWARE[key]["cost_per_hr_usd"]
    if axis == "decode_tps":
        hw, model, prec = key.split("|")
        return tokens_per_sec(Tier(hw, model, prec))
    if axis == "quality":
        use_case, model = key.split("|")
        return BASE_QUALITY[use_case][model]
    if axis == "quant_drop":
        _, prec = key.split("|")
        return PRECISION_QUALITY_PENALTY[prec]
    raise ValueError(axis)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", default=str(ROOT / "data/benchmarks/external_references_20261127.csv"))
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.refs)))
    out, by_axis = [], {}
    for r in rows:
        ref, pred = float(r["reference"]), predict(r["axis"], r["sim_key"])
        if r["axis"] == "quant_drop":
            err, ok, shown = None, abs(pred - ref) <= TOL_ABS, f"{pred - ref:+.3f} abs"
        else:
            err = 100 * (pred - ref) / ref
            ok, shown = abs(err) <= TOL_PCT, f"{err:+.0f}%"
        by_axis.setdefault(r["axis"], []).append(ok)
        out.append((r["axis"], r["sim_key"], pred, ref, r["units"], shown, "PASS" if ok else "FAIL",
                    r["source_name"], (pred / ref) if r["axis"] == "decode_tps" else None))
    print("| Axis | Simulator key | Simulator | Real (sourced) | Units | Error | Gate | Source |")
    print("|---|---|---|---|---|---|---|---|")
    for a, k, p, ref, u, e, g, s, _ in out:
        print(f"| {a} | `{k}` | {p:.3g} | {ref:.3g} | {u} | {e} | {g} | {s} |")
    print()
    for a, oks in by_axis.items():
        print(f"- **{a}**: {sum(oks)}/{len(oks)} references within gate")
    ratios = [x[-1] for x in out if x[-1] is not None]
    if ratios:
        print(f"- decode speed: simulator is {min(ratios):.1f}x to {max(ratios):.1f}x faster than measured "
              f"(median {statistics.median(ratios):.1f}x)")


if __name__ == "__main__":
    main()
