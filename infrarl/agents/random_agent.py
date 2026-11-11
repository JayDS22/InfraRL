"""Uniform random over the legal (masked) action set. Lower bound."""
from __future__ import annotations

import numpy as np

from .base import BaseAgent


class RandomAgent(BaseAgent):
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def select_action(self, obs, mask, explore=True) -> int:
        return int(self.rng.choice(np.flatnonzero(mask)))
