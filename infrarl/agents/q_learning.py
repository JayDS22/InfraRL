"""Tabular Q-learning over a discretized (profile, partial-stack) state.

The sequential MDP has up to 11 steps, each with a Discrete(18) action space
and a per-step mask. The continuous profile features are bucketed into a
coarse hash key so the table stays small (~10^4-10^5 entries). Missing
(state, action) entries default to 0.0 (optimistic init at the max reachable
reward would overshoot and never learn, so we stay at 0 and let epsilon
drive early exploration).

Update rule (Sutton & Barto 6.5): Q(s, a) <- Q(s, a) + alpha * (target - Q(s, a))
with target = r_t + gamma * max_a'[Q(s', a')] over the legal a', or r_T at
the terminal step.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..simulator.environment import N_ACTIONS
from ..simulator.models import USE_CASES
from ..simulator.profiles import SENSITIVITY
from .base import BaseAgent, Transition


def _bucket(profile) -> tuple:
    """Coarse hash of a profile for table keying: 5 quantized features + 2 categoricals."""
    return (
        profile.use_case,
        profile.data_sensitivity,
        int(np.log10(max(profile.monthly_queries, 1)) * 2),  # ~14 buckets
        int(np.log10(max(profile.latency_sla_ms, 1)) * 2),   # ~10 buckets
        int(profile.acc_req * 10),                           # 6-9 buckets
        int(np.log10(max(profile.monthly_budget_usd, 1)) * 2),  # ~10 buckets
        int(profile.peak_multiplier),                        # 1-5
    )


def _stack_key(tiers: list, scaling: str | None) -> tuple:
    """Compact key for the partial stack so similar partials share a Q row."""
    return (tuple((t[0], t[1], t[2]) for t in tiers), scaling)


class QLearningAgent(BaseAgent):
    name = "q_learning"

    def __init__(
        self,
        seed: int = 0,
        lr: float = 0.2,
        gamma: float = 1.0,  # terminal-only reward, no intermediate discount needed
        epsilon_start: float = 0.5,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 5000,
    ):
        self.rng = np.random.default_rng(seed)
        self.lr = lr
        self.gamma = gamma
        self.eps_start = epsilon_start
        self.eps_end = epsilon_end
        self.eps_decay = epsilon_decay_episodes
        self.Q: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(N_ACTIONS))
        self._episodes = 0
        self._profile_key: tuple | None = None
        self._history: tuple = ()  # actions chosen this episode
        self._pending: tuple | None = None  # (state_key, action)

    def _epsilon(self) -> float:
        frac = min(1.0, self._episodes / max(1, self.eps_decay))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def begin_episode(self, profile) -> None:
        self._profile_key = _bucket(profile)
        self._history = ()
        self._pending = None

    def select_action(self, obs: np.ndarray, mask: np.ndarray, explore: bool = True) -> int:
        s = (self._profile_key, self._history)
        legal = np.flatnonzero(mask)
        if explore and self.rng.random() < self._epsilon():
            a = int(self.rng.choice(legal))
        else:
            q = self.Q[s]
            masked = np.where(mask, q, -np.inf)
            best = np.flatnonzero(masked == masked.max())
            a = int(self.rng.choice(best))
        self._pending = (s, a)
        return a

    def update(self, t: Transition) -> None:
        if self._pending is None:
            return
        s, a = self._pending
        new_history = self._history + (a,)
        if t.terminated:
            target = t.reward
            self._episodes += 1
        else:
            s_next = (self._profile_key, new_history)
            q_next = self.Q[s_next]
            legal_next = np.where(t.next_mask, q_next, -np.inf)
            target = t.reward + self.gamma * float(legal_next.max()) if t.next_mask.any() else t.reward
        self.Q[s][a] += self.lr * (target - self.Q[s][a])
        self._history = new_history
        self._pending = None
