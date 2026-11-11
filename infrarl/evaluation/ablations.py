"""Ablations from the proposal. Each one removes a decision from the agent by
restricting the env's action mask (so any mask-aware agent adapts for free) and,
for arm-based bandits, filtering the playable arms the same way.

    no_hardware_awareness  hardware fixed to A100 (does cost go up?)
    no_precision_tuning    precision fixed to fp16, falling back to int8/int4 only
                           when fp16 does not fit (does quality drop?)
    no_scaling_policy      scaling fixed to static (do SLA violations spike at peaks?)
    no_multitier           single-tier stacks only
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..simulator.environment import (
    ARCH_RANGE, HW_RANGE, MODEL_RANGE, PREC_RANGE, SCALE_RANGE, STEP_ARCH, STEP_HW,
    STEP_PREC, STEP_SCALE, InfraEnv,
)
from ..simulator.hardware import HARDWARE, HARDWARE_NAMES
from ..simulator.models import MODEL_NAMES, PRECISION_NAMES, min_memory_gb

_PREC_PREFERENCE = ["fp16", "int8", "int4"]


@dataclass(frozen=True)
class Ablation:
    name: str
    fixed_hardware: str | None = None
    fp16_only: bool = False
    static_scaling: bool = False
    single_tier: bool = False

    def restrict(self, step_type: int, mask: np.ndarray, hw: str | None = None, model: str | None = None) -> np.ndarray:
        m = mask.copy()
        if step_type == STEP_ARCH and self.single_tier:
            m[ARCH_RANGE.start + 1 : ARCH_RANGE.stop] = False
        elif step_type == STEP_HW and self.fixed_hardware:
            m[HW_RANGE.start : HW_RANGE.stop] = False
            m[HW_RANGE.start + HARDWARE_NAMES.index(self.fixed_hardware)] = True
        elif step_type == STEP_PREC and self.fp16_only:
            keep = next(p for p in _PREC_PREFERENCE if mask[PREC_RANGE.start + PRECISION_NAMES.index(p)])
            m[PREC_RANGE.start : PREC_RANGE.stop] = False
            m[PREC_RANGE.start + PRECISION_NAMES.index(keep)] = True
        elif step_type == STEP_SCALE and self.static_scaling:
            m[SCALE_RANGE.start + 1 : SCALE_RANGE.stop] = False
        return m

    def arm_ok(self, actions: list[int]) -> bool:
        """actions = [arch, hw, model, precision, scaling] of a single-tier arm."""
        _, hw, model, prec, scale = actions
        hw_n = HARDWARE_NAMES[hw - HW_RANGE.start]
        model_n = MODEL_NAMES[model - MODEL_RANGE.start]
        if self.fixed_hardware and hw_n != self.fixed_hardware:
            return False
        if self.fp16_only:
            fits = [p for p in _PREC_PREFERENCE if min_memory_gb(model_n, p) <= HARDWARE[hw_n]["memory_gb"]]
            if PRECISION_NAMES[prec - PREC_RANGE.start] != fits[0]:
                return False
        if self.static_scaling and scale != SCALE_RANGE.start:
            return False
        return True


ABLATIONS = {
    a.name: a
    for a in [
        Ablation("no_hardware_awareness", fixed_hardware="a100"),
        Ablation("no_precision_tuning", fp16_only=True),
        Ablation("no_scaling_policy", static_scaling=True),
        Ablation("no_multitier", single_tier=True),
    ]
}


class AblatedInfraEnv(InfraEnv):
    def __init__(self, ablation: Ablation, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ablation = ablation

    def action_masks(self) -> np.ndarray:
        mask = super().action_masks()
        if self._done:
            return mask
        step_type, _ = self._schedule[self._t]
        return self.ablation.restrict(step_type, mask)
