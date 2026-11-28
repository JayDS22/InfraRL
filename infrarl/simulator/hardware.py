"""Hardware lookup table, recalibrated 2026-11-27 against public data.

Sources and fit: docs/EXTERNAL_REFERENCE_CHECK.md, data/benchmarks/external_references_20261127.csv.
- cost_per_hr_usd: kept where already within 15% of most sources (A10 = AWS g5.xlarge
  $1.006; H100 = Nebius $4.50, Lambda $4.59, Runpod $4.39); otherwise the median of
  sourced on-demand prices (A100 80GB: AWS/8, Lambda, Runpod SXM, Runpod PCIe -> $2.69;
  CPU 128 GiB: AWS r7i.4xlarge $1.06).
- tps_13b_fp16: least-squares fit (log space) of the estimator's speed formula to measured
  llama.cpp batch-1 decode speeds for Llama 3 8B/70B on A100 SXM 80GB and H100 PCIe 80GB.
  A10 and CPU have no public measured reference; they keep the original placeholder ratio
  to A100 (0.36 and 0.03) and are UNVERIFIED.
"""
from __future__ import annotations

HARDWARE_NAMES = ("cpu", "a10", "a100", "h100")

HARDWARE = {
    # tps_13b_fp16: single-stream decode tokens/sec for a 13B model at fp16
    "cpu": {"memory_gb": 128, "cost_per_hr_usd": 1.06, "tps_13b_fp16": 0.92},   # speed unverified
    "a10": {"memory_gb": 24, "cost_per_hr_usd": 1.00, "tps_13b_fp16": 11.0},    # speed unverified
    "a100": {"memory_gb": 80, "cost_per_hr_usd": 2.69, "tps_13b_fp16": 30.5},
    "h100": {"memory_gb": 80, "cost_per_hr_usd": 4.50, "tps_13b_fp16": 34.3},
}

HOURS_PER_MONTH = 24 * 30
