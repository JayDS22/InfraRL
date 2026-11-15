"""Shared Agent protocol. All four methods (bandit, Q-learning, DQN, PPO) and the
baselines implement this so the evaluation runner has a single code path."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    next_mask: np.ndarray
    terminated: bool
    # Optional step info dict. Populated by run_episode; carries quality,
    # latency_ms, cost_usd, meets_quality, within_budget, meets_sla, stack on
    # terminals. Used by constrained-RL variants that need per-episode
    # compliance signals beyond the scalar reward (e.g., LagrangianPPOAgent).
    info: dict = field(default_factory=dict)


@runtime_checkable
class Agent(Protocol):
    name: str

    def begin_episode(self, profile) -> None:
        """Optional hook; called after env.reset(). Learned agents may ignore it."""

    def select_action(self, obs: np.ndarray, mask: np.ndarray, explore: bool = True) -> int:
        """Return a legal action (mask[a] is True)."""

    def update(self, transition: Transition) -> None:
        """Called after every env step (reward is 0 until the terminal step)."""


class BaseAgent:
    name = "base"

    def begin_episode(self, profile) -> None:
        pass

    def update(self, transition: Transition) -> None:
        pass

    def select_action(self, obs, mask, explore=True) -> int:
        raise NotImplementedError
