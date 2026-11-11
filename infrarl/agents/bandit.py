"""Bandit baselines over single-tier full-stack configurations.

Each arm is one (hardware, model, precision, scaling) single-tier stack; pulling
an arm plays its 5 actions through the env and the terminal reward updates it.

* ContextFreeBandit: epsilon-greedy with sample-mean values; ignores the profile.
* ContextualBandit: LinUCB, one ridge model per arm over the profile features.
"""
from __future__ import annotations

import itertools

import numpy as np

from ..simulator.environment import (
    ARCH_RANGE, HW_RANGE, MODEL_RANGE, N_PROFILE_FEATURES, PREC_RANGE, SCALE_RANGE,
)
from ..simulator.estimator import SCALING_POLICIES, Tier, is_feasible
from ..simulator.hardware import HARDWARE_NAMES
from ..simulator.models import MODEL_NAMES, PRECISION_NAMES
from .base import BaseAgent, Transition


def enumerate_arms() -> list[list[int]]:
    arms = []
    for h, m, p, s in itertools.product(range(4), range(4), range(4), range(3)):
        if is_feasible(Tier(HARDWARE_NAMES[h], MODEL_NAMES[m], PRECISION_NAMES[p])):
            arms.append([ARCH_RANGE.start, HW_RANGE.start + h, MODEL_RANGE.start + m,
                         PREC_RANGE.start + p, SCALE_RANGE.start + s])
    return arms


class _ArmBandit(BaseAgent):
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.arms = enumerate_arms()
        self.arm_allowed = np.ones(len(self.arms), dtype=bool)
        self._queue: list[int] = []
        self._arm = -1
        self._ctx: np.ndarray | None = None

    def set_arm_filter(self, fn) -> None:
        """Restrict playable arms (used by ablations). fn(actions: list[int]) -> bool."""
        self.arm_allowed = np.array([bool(fn(a)) for a in self.arms]) if fn else np.ones(len(self.arms), dtype=bool)

    def _choose(self, ctx: np.ndarray, explore: bool) -> int:
        raise NotImplementedError

    def _learn(self, arm: int, ctx: np.ndarray, reward: float) -> None:
        raise NotImplementedError

    def select_action(self, obs, mask, explore=True) -> int:
        if not self._queue:  # first step of an episode: pick an arm from the profile
            self._ctx = np.append(obs[:N_PROFILE_FEATURES], 1.0).astype(np.float64)
            self._arm = self._choose(self._ctx, explore)
            self._queue = list(self.arms[self._arm])
        return self._queue.pop(0)

    def update(self, t: Transition) -> None:
        if t.terminated and self._ctx is not None:
            self._learn(self._arm, self._ctx, t.reward)


class ContextFreeBandit(_ArmBandit):
    name = "bandit_context_free"

    def __init__(self, epsilon: float = 0.1, seed: int = 0):
        super().__init__(seed)
        self.epsilon = epsilon
        self.values = np.ones(len(self.arms))  # optimistic init: tries every arm once
        self.counts = np.zeros(len(self.arms))

    def _choose(self, ctx, explore):
        allowed = np.flatnonzero(self.arm_allowed)
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.choice(allowed))
        v = np.where(self.arm_allowed, self.values, -np.inf)
        return int(self.rng.choice(np.flatnonzero(v == v.max())))

    def _learn(self, arm, ctx, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] = reward if n == 1 else self.values[arm] + (reward - self.values[arm]) / n


class ContextualBandit(_ArmBandit):
    name = "bandit_contextual"

    def __init__(self, alpha: float = 0.3, seed: int = 0):
        super().__init__(seed)
        d = N_PROFILE_FEATURES + 1
        self.alpha = alpha
        self.A_inv = np.stack([np.eye(d) for _ in self.arms])
        self.b = np.zeros((len(self.arms), d))

    def _choose(self, ctx, explore):
        theta = np.einsum("aij,aj->ai", self.A_inv, self.b)
        mean = theta @ ctx
        if explore:
            mean = mean + self.alpha * np.sqrt(np.einsum("i,aij,j->a", ctx, self.A_inv, ctx))
        return int(np.argmax(np.where(self.arm_allowed, mean, -np.inf)))

    def _learn(self, arm, ctx, reward):
        Ai = self.A_inv[arm]
        Ax = Ai @ ctx
        self.A_inv[arm] = Ai - np.outer(Ax, Ax) / (1.0 + ctx @ Ax)  # Sherman-Morrison
        self.b[arm] += reward * ctx
