"""Agent registry: the single place the evaluation harness looks up methods.

To plug in a new method (Q-learning, DQN, PPO, ...):

    1. Implement the Agent protocol (agents/base.py): begin_episode, select_action
       (must return a legal action under `mask`), update(transition).
    2. Register it here (or call `register(...)` from your own module):

           register("dqn", lambda seed: DQNAgent(seed=seed), train_epochs=30)

    3. python -m infrarl.evaluation.run_experiment --agents dqn,heuristic,random
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable

from .bandit import ContextFreeBandit, ContextualBandit
from .heuristic import HeuristicAgent
from .random_agent import RandomAgent


@dataclass
class AgentSpec:
    name: str
    factory: Callable[[int], object]  # seed -> agent
    train_epochs: int = 0  # passes over the training profiles (0 = no learning)


REGISTRY: dict[str, AgentSpec] = {}


def register(name: str, factory: Callable[[int], object], train_epochs: int = 0) -> None:
    REGISTRY[name] = AgentSpec(name, factory, train_epochs)


register("random", lambda s: RandomAgent(s))
register("heuristic", lambda s: HeuristicAgent())
register("bandit_context_free", lambda s: ContextFreeBandit(seed=s), train_epochs=20)
register("bandit_contextual", lambda s: ContextualBandit(seed=s), train_epochs=20)


def _try_register(name: str, module: str, cls: str, epochs: int) -> None:
    """Auto-register teammates' agents once their class exists (empty files are skipped)."""
    try:
        agent_cls = getattr(importlib.import_module(f"{__package__}.{module}"), cls)
    except (ImportError, AttributeError):
        return
    register(name, lambda s, c=agent_cls: c(seed=s), train_epochs=epochs)


_try_register("q_learning", "q_learning", "QLearningAgent", 30)
_try_register("dqn", "dqn_agent", "DQNAgent", 30)
_try_register("ppo", "ppo_agent", "PPOAgent", 30)
