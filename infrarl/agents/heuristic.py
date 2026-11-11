"""Rule-based baseline approximating a cost-aware cloud sales engineer.

Single-tier. Picks the smallest model whose fp16 quality for the use case meets
the accuracy requirement (largest model if none does), then the cheapest GPU on
which that model fits at fp16 (dropping to a smaller model if none fits), fp16
precision, and conservative autoscaling.
"""
from __future__ import annotations

from ..simulator.environment import (
    ARCH_RANGE, HW_RANGE, MODEL_RANGE, PREC_RANGE, SCALE_RANGE,
)
from ..simulator.hardware import HARDWARE, HARDWARE_NAMES
from ..simulator.models import BASE_QUALITY, MODEL_NAMES, PRECISION_NAMES, min_memory_gb
from ..simulator.estimator import SCALING_POLICIES
from .base import BaseAgent

_GPUS = [h for h in HARDWARE_NAMES if h != "cpu"]
_GPUS.sort(key=lambda h: HARDWARE[h]["cost_per_hr_usd"])


def plan(profile) -> list[int]:
    q = BASE_QUALITY[profile.use_case]
    idx = next((i for i, m in enumerate(MODEL_NAMES) if q[m] >= profile.acc_req), len(MODEL_NAMES) - 1)
    while True:
        model = MODEL_NAMES[idx]
        hw = next((h for h in _GPUS if min_memory_gb(model, "fp16") <= HARDWARE[h]["memory_gb"]), None)
        if hw is not None or idx == 0:
            break
        idx -= 1
    return [
        ARCH_RANGE.start,
        HW_RANGE.start + HARDWARE_NAMES.index(hw),
        MODEL_RANGE.start + idx,
        PREC_RANGE.start + PRECISION_NAMES.index("fp16"),
        SCALE_RANGE.start + SCALING_POLICIES.index("conservative"),
    ]


class HeuristicAgent(BaseAgent):
    name = "heuristic"

    def __init__(self):
        self._queue: list[int] = []

    def begin_episode(self, profile) -> None:
        self._queue = plan(profile)

    def select_action(self, obs, mask, explore=True) -> int:
        a = self._queue.pop(0)
        assert mask[a], f"heuristic planned a masked action {a}"
        return a
