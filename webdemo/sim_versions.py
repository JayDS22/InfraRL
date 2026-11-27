"""Two simulator constant sets the demo can switch between.

- "placeholder": the original tables (main as of 2026-11-26, results_frozen_v2_test).
- "calibrated": refit against 29 public references (PR #5, docs/EXTERNAL_REFERENCE_CHECK.md,
  results/recalibrated_20261127).

`apply(name)` rewrites the simulator's lookup dicts in place. The estimator reads them at
call time, so every later estimate uses the chosen set. Callers must hold a lock while
applying and computing, so two requests never mix constant sets.
"""
from __future__ import annotations

import copy

from infrarl.simulator import hardware, models

_SHARED_QUALITY = {
    "batch_inference": {"3b": 0.60, "7b": 0.72, "13b": 0.80, "70b": 0.90},
    "rag": {"3b": 0.58, "7b": 0.70, "13b": 0.80, "70b": 0.91},
    "agentic": {"3b": 0.40, "7b": 0.58, "13b": 0.72, "70b": 0.90},
    "structured_extraction": {"3b": 0.65, "7b": 0.77, "13b": 0.85, "70b": 0.93},
}

SIMS = {
    "placeholder": {
        "label": "Placeholder numbers",
        "hardware": {
            "cpu": {"memory_gb": 128, "cost_per_hr_usd": 0.40, "tps_13b_fp16": 15},
            "a10": {"memory_gb": 24, "cost_per_hr_usd": 1.00, "tps_13b_fp16": 180},
            "a100": {"memory_gb": 80, "cost_per_hr_usd": 2.40, "tps_13b_fp16": 500},
            "h100": {"memory_gb": 80, "cost_per_hr_usd": 4.50, "tps_13b_fp16": 1100},
        },
        "precision_speed": {"int4": 2.2, "int8": 1.6, "fp16": 1.0, "fp32": 0.5},
        "precision_penalty": {"int4": 0.04, "int8": 0.01, "fp16": 0.0, "fp32": 0.0},
        "base_quality": {
            "chatbot": {"3b": 0.62, "7b": 0.76, "13b": 0.85, "70b": 0.95},
            "code_generation": {"3b": 0.45, "7b": 0.62, "13b": 0.74, "70b": 0.92},
            **_SHARED_QUALITY,
        },
        # official 3-seed test-set results under these constants (results_frozen_v2_test)
        "official": {"lagrangian_ppo": 0.650, "ppo": 0.484, "heuristic": 0.460,
                     "bandit_contextual": 0.376, "random": 0.136},
    },
    "calibrated": {
        "label": "Real data (calibrated)",
        "hardware": {
            "cpu": {"memory_gb": 128, "cost_per_hr_usd": 1.06, "tps_13b_fp16": 0.92},
            "a10": {"memory_gb": 24, "cost_per_hr_usd": 1.00, "tps_13b_fp16": 11.0},
            "a100": {"memory_gb": 80, "cost_per_hr_usd": 2.69, "tps_13b_fp16": 30.5},
            "h100": {"memory_gb": 80, "cost_per_hr_usd": 4.50, "tps_13b_fp16": 34.3},
        },
        "precision_speed": {"int4": 3.08, "int8": 1.76, "fp16": 1.0, "fp32": 0.5},
        "precision_penalty": {"int4": 0.021, "int8": 0.01, "fp16": 0.0, "fp32": 0.0},
        "base_quality": {
            "chatbot": {"3b": 0.644, "7b": 0.754, "13b": 0.800, "70b": 0.868},
            "code_generation": {"3b": 0.736, "7b": 0.820, "13b": 0.828, "70b": 0.874},
            **_SHARED_QUALITY,
        },
        # official 3-seed test-set results under these constants (results/recalibrated_20261127)
        "official": {"lagrangian_ppo": 0.068, "ppo": 0.089, "heuristic": 0.049,
                     "bandit_contextual": 0.161, "random": 0.022},
    },
}


def _replace(target: dict, source: dict) -> None:
    target.clear()
    target.update(copy.deepcopy(source))


def apply(name: str) -> None:
    s = SIMS[name]
    _replace(hardware.HARDWARE, s["hardware"])
    _replace(models.PRECISION_SPEED, s["precision_speed"])
    _replace(models.PRECISION_QUALITY_PENALTY, s["precision_penalty"])
    _replace(models.BASE_QUALITY, s["base_quality"])
