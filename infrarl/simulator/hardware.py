"""Hardware lookup table.

Numbers are rough estimates from Nebius Token Factory published specs and NVIDIA
MLPerf results. They are PLACEHOLDERS until the 3-config residual check
(report Section 2.2) is run; see simulator/estimator.py.
"""
from __future__ import annotations

HARDWARE_NAMES = ("cpu", "a10", "a100", "h100")

HARDWARE = {
    # tps_13b_fp16: single-stream decode tokens/sec for a 13B model at fp16
    "cpu": {"memory_gb": 128, "cost_per_hr_usd": 0.40, "tps_13b_fp16": 15},
    "a10": {"memory_gb": 24, "cost_per_hr_usd": 1.00, "tps_13b_fp16": 180},
    "a100": {"memory_gb": 80, "cost_per_hr_usd": 2.40, "tps_13b_fp16": 500},
    "h100": {"memory_gb": 80, "cost_per_hr_usd": 4.50, "tps_13b_fp16": 1100},
}

HOURS_PER_MONTH = 24 * 30
