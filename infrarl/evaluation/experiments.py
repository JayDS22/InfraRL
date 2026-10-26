"""Experiment runner: one code path for every agent."""
from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np

from ..agents.base import Transition
from ..simulator.environment import InfraEnv
from ..simulator.profiles import Profile


@dataclass
class EpisodeResult:
    profile: Profile
    quality: float
    latency_ms: float
    cost_usd: float
    reward: float
    within_budget: bool
    meets_sla: bool
    meets_quality: bool
    tier_count: int
    stack: str

    @property
    def meets_all(self) -> bool:
        return self.within_budget and self.meets_sla and self.meets_quality


def run_episode(env: InfraEnv, agent, profile: Profile, train: bool = True) -> EpisodeResult:
    obs, info = env.reset(options={"profile": profile})
    agent.begin_episode(profile)
    mask = info["action_mask"]
    while True:
        a = agent.select_action(obs, mask, explore=train)
        nobs, r, term, trunc, info = env.step(a)
        if train:
            agent.update(Transition(obs, a, r, nobs, info["action_mask"], term))
        obs, mask = nobs, info["action_mask"]
        if term or trunc:
            break
    s = info["stack"]
    desc = "|".join(f"{t.hardware}-{t.model}-{t.precision}" for t in s.tiers) + f"|{s.scaling}"
    return EpisodeResult(profile, info["quality"], info["latency_ms"], info["cost_usd"], r,
                         info["within_budget"], info["meets_sla"], info["quality"] >= profile.acc_req,
                         s.tier_count, desc)


def train(env: InfraEnv, agent, profiles: list[Profile], epochs: int, seed: int = 0,
          eval_profiles: list[Profile] | None = None) -> list[dict]:
    """Train for `epochs` passes over shuffled profiles. Returns a learning curve:
    one row per epoch with mean training reward (exploratory) and, if eval_profiles
    is given, mean greedy reward on the held-out set."""
    rng = np.random.default_rng(seed)
    curve = []
    for ep in range(epochs):
        rewards = [run_episode(env, agent, profiles[i], train=True).reward for i in rng.permutation(len(profiles))]
        row = {"epoch": ep + 1, "train_reward": float(np.mean(rewards))}
        if eval_profiles is not None:
            row["eval_reward"] = float(np.mean([r.reward for r in evaluate(env, agent, eval_profiles)]))
        curve.append(row)
    return curve


def evaluate(env: InfraEnv, agent, profiles: list[Profile], train_epochs: int = 0,
             seed: int = 0) -> list[EpisodeResult]:
    """Optionally train for train_epochs passes, then run one greedy pass (no
    exploration, no updates) over all profiles."""
    if train_epochs:
        train(env, agent, profiles, train_epochs, seed)
    return [run_episode(env, agent, p, train=False) for p in profiles]


def summarize(results: list[EpisodeResult]) -> dict:
    """Operational metrics. `avg_quality` is the proposal's metric (raw quality);
    `quality_attainment` is quality / requirement, clipped at 1, which is
    comparable across profiles with different accuracy requirements."""
    return {
        "avg_quality": float(np.mean([r.quality for r in results])),
        "quality_attainment": float(np.mean([min(1.0, r.quality / r.profile.acc_req) for r in results])),
        "quality_compliance": float(np.mean([r.meets_quality for r in results])),
        "budget_compliance": float(np.mean([r.within_budget for r in results])),
        "sla_compliance": float(np.mean([r.meets_sla for r in results])),
        "meets_all": float(np.mean([r.meets_all for r in results])),
        "mean_reward": float(np.mean([r.reward for r in results])),
        "mean_cost_usd": float(np.mean([r.cost_usd for r in results])),
    }


EPISODE_FIELDS = ["agent", "seed", "profile_idx", "use_case", "acc_req", "budget", "sla_ms", "quality",
                  "latency_ms", "cost_usd", "reward", "within_budget", "meets_sla", "meets_quality",
                  "tier_count", "stack"]


def episode_rows(agent: str, seed: int, results: list[EpisodeResult]):
    for i, r in enumerate(results):
        yield [agent, seed, i, r.profile.use_case, f"{r.profile.acc_req:.4f}",
               f"{r.profile.monthly_budget_usd:.1f}", f"{r.profile.latency_sla_ms:.1f}",
               f"{r.quality:.4f}", f"{r.latency_ms:.1f}", f"{r.cost_usd:.1f}", f"{r.reward:.4f}",
               int(r.within_budget), int(r.meets_sla), int(r.meets_quality), r.tier_count, r.stack]


def write_episodes(path: str, rows) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(EPISODE_FIELDS)
        w.writerows(rows)
