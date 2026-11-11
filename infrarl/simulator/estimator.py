"""Lookup-table estimator: (stack, profile) -> (quality, latency_ms, cost_usd_per_month).

Multi-tier stacks are modelled as a cascade/router: tier i serves a fixed
fraction of traffic. Quality and latency are traffic-weighted averages; cost is
the sum over tiers of replicas * hourly price.

NOTE: constants here are uncalibrated placeholders. The 3-config residual check
(Small CPU+3B+int8, Mid A10+13B+fp16, Large A100+70B+fp16; <=15% per axis) must
pass before reported numbers are trusted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .hardware import HARDWARE, HOURS_PER_MONTH
from .models import (
    BASE_QUALITY,
    PRECISION_QUALITY_PENALTY,
    PRECISION_SPEED,
    PARAMS_B,
    min_memory_gb,
)
from .profiles import Profile

SCALING_POLICIES = ("static", "conservative", "aggressive")
# multiple of average QPS provisioned (None = provision for peak)
_PROVISION_FACTOR = {"static": None, "conservative": 1.5, "aggressive": 1.1}
# flat latency overhead from scale-up delay
_SCALING_LATENCY = {"static": 1.0, "conservative": 1.05, "aggressive": 1.15}
TRAFFIC_SPLIT = {1: (1.0,), 2: (0.7, 0.3), 3: (0.5, 0.3, 0.2)}
BATCH_GAIN = 4.0  # concurrent requests a replica serves via continuous batching


@dataclass(frozen=True)
class Tier:
    hardware: str
    model: str
    precision: str


@dataclass(frozen=True)
class Stack:
    tiers: tuple[Tier, ...]
    scaling: str

    @property
    def tier_count(self) -> int:
        return len(self.tiers)


def is_feasible(tier: Tier) -> bool:
    return min_memory_gb(tier.model, tier.precision) <= HARDWARE[tier.hardware]["memory_gb"]


def tokens_per_sec(tier: Tier) -> float:
    return (
        HARDWARE[tier.hardware]["tps_13b_fp16"]
        * (13.0 / PARAMS_B[tier.model])
        * PRECISION_SPEED[tier.precision]
    )


def tier_quality(tier: Tier, use_case: str) -> float:
    return BASE_QUALITY[use_case][tier.model] - PRECISION_QUALITY_PENALTY[tier.precision]


def estimate(stack: Stack, profile: Profile) -> tuple[float, float, float]:
    """Return (quality in [0,1], mean latency in ms, cost in USD/month)."""
    split = TRAFFIC_SPLIT[stack.tier_count]
    factor = _PROVISION_FACTOR[stack.scaling]
    provisioned_qps = (
        profile.peak_qps
        if factor is None
        else min(profile.peak_qps, profile.avg_qps * factor)
    )
    quality = latency = cost = 0.0
    for tier, frac in zip(stack.tiers, split):
        if not is_feasible(tier):
            raise ValueError(f"infeasible tier {tier}")
        tps = tokens_per_sec(tier)
        capacity_qps = tps * BATCH_GAIN / profile.avg_output_tokens
        replicas = max(1, math.ceil(provisioned_qps * frac / capacity_qps))
        rho_peak = min(3.0, profile.peak_qps * frac / (replicas * capacity_qps))
        base_ms = profile.avg_output_tokens / tps * 1000.0
        lat = base_ms * (1.0 + 2.0 * max(0.0, rho_peak - 0.6)) * _SCALING_LATENCY[stack.scaling]
        quality += frac * tier_quality(tier, profile.use_case)
        latency += frac * lat
        cost += replicas * HARDWARE[tier.hardware]["cost_per_hr_usd"] * HOURS_PER_MONTH
    return quality, latency, cost
