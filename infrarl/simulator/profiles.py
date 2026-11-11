"""Customer profiles and the synthetic profile generator."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import USE_CASES

SENSITIVITY = ("low", "medium", "high")

# (lo, hi) ranges per use case: avg output tokens, avg input tokens, latency SLA ms
_USE_CASE_SPECS = {
    "chatbot": {"out": (100, 400), "inp": (50, 500), "sla": (400, 2000)},
    "code_generation": {"out": (150, 600), "inp": (200, 2000), "sla": (800, 4000)},
    "batch_inference": {"out": (50, 300), "inp": (100, 1000), "sla": (5000, 30000)},
    "rag": {"out": (100, 300), "inp": (1000, 4000), "sla": (700, 3000)},
    "agentic": {"out": (200, 800), "inp": (500, 3000), "sla": (1500, 8000)},
    "structured_extraction": {"out": (30, 150), "inp": (200, 2000), "sla": (300, 1500)},
}


@dataclass(frozen=True)
class Profile:
    use_case: str
    monthly_queries: float
    latency_sla_ms: float
    acc_req: float
    monthly_budget_usd: float
    data_sensitivity: str
    peak_multiplier: float
    avg_input_tokens: float
    avg_output_tokens: float
    concurrent_users: float

    @property
    def avg_qps(self) -> float:
        return self.monthly_queries / (30 * 24 * 3600)

    @property
    def peak_qps(self) -> float:
        return self.avg_qps * self.peak_multiplier


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def sample_profile(rng: np.random.Generator) -> Profile:
    use_case = USE_CASES[int(rng.integers(len(USE_CASES)))]
    spec = _USE_CASE_SPECS[use_case]
    return Profile(
        use_case=use_case,
        monthly_queries=_log_uniform(rng, 1e5, 3e7),
        latency_sla_ms=_log_uniform(rng, *spec["sla"]),
        acc_req=float(rng.uniform(0.60, 0.90)),
        monthly_budget_usd=_log_uniform(rng, 1_000, 30_000),
        data_sensitivity=SENSITIVITY[int(rng.integers(3))],
        peak_multiplier=float(rng.uniform(1.5, 5.0)),
        avg_input_tokens=_log_uniform(rng, *spec["inp"]),
        avg_output_tokens=_log_uniform(rng, *spec["out"]),
        concurrent_users=_log_uniform(rng, 10, 5_000),
    )


def generate_profiles(n: int = 1000, seed: int = 0) -> list[Profile]:
    """Deterministic list of n synthetic profiles across the six use cases."""
    rng = np.random.default_rng(seed)
    return [sample_profile(rng) for _ in range(n)]
