"""Workload drift simulator + online-vs-static adaptation experiment.

Proposal Section 5.1 promised: "Simulate workload drift over 4 weeks.
Measure how quickly the contextual bandit adapts vs a static recommendation."

Operationalization: a profile stream with a linearly interpolated use-case
mix. Week 1 is the training distribution Vishnu's bandit was fit to
(uniform across the six use cases). Week N shifts toward a target
distribution (chatbot + rag heavy, mimicking a product-launch spike).

Static evaluator: train once on pre-drift profiles, run greedy inference
on the drift stream without updating.
Online evaluator: same agent, keeps exploring + learning across the stream.

Metric: mean reward per week, cumulative regret vs the brute-force oracle
per profile (optional, expensive). The adaptation claim is "online catches
back up to static within W weeks of drift onset."
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from ..evaluation.experiments import run_episode
from ..simulator.environment import InfraEnv
from ..simulator.models import USE_CASES
from ..simulator.profiles import SENSITIVITY, Profile, _USE_CASE_SPECS, _log_uniform


_BASE_ACC_REQ = (0.60, 0.90)
_SHIFTED_ACC_REQ = (0.75, 0.95)       # later-week acc_req pulled toward the hard end
_BASE_BUDGET = (1_000.0, 30_000.0)
_SHIFTED_BUDGET = (500.0, 15_000.0)   # half the money, same workload


def drift_mix(weeks: int, pre_drift_weeks: int = 1) -> np.ndarray:
    """Per-week probability mass over USE_CASES.

    Pre-drift weeks are uniform. After that, linear interpolation to a
    target mix where chatbot and rag jointly take 70% of traffic.
    """
    base = np.ones(len(USE_CASES)) / len(USE_CASES)
    target = np.full(len(USE_CASES), 0.3 / (len(USE_CASES) - 2))
    target[USE_CASES.index("chatbot")] = 0.4
    target[USE_CASES.index("rag")] = 0.3
    mix = np.zeros((weeks, len(USE_CASES)))
    for w in range(weeks):
        if w < pre_drift_weeks:
            mix[w] = base
        else:
            drift = (weeks - 1 - pre_drift_weeks)
            alpha = min(1.0, (w - pre_drift_weeks + 1) / max(1, drift))
            mix[w] = (1 - alpha) * base + alpha * target
    return mix


def _week_ranges(weeks: int, pre_drift_weeks: int, drift_acc_req: bool,
                 drift_budget: bool) -> tuple[list[tuple], list[tuple]]:
    """Per-week (acc_req_range, budget_range), linearly interpolated from base
    to shifted ranges after the pre-drift window."""
    acc, bud = [], []
    for w in range(weeks):
        if w < pre_drift_weeks:
            alpha = 0.0
        else:
            drift = (weeks - 1 - pre_drift_weeks)
            alpha = min(1.0, (w - pre_drift_weeks + 1) / max(1, drift))
        if drift_acc_req:
            lo = (1 - alpha) * _BASE_ACC_REQ[0] + alpha * _SHIFTED_ACC_REQ[0]
            hi = (1 - alpha) * _BASE_ACC_REQ[1] + alpha * _SHIFTED_ACC_REQ[1]
            acc.append((lo, hi))
        else:
            acc.append(_BASE_ACC_REQ)
        if drift_budget:
            lo = (1 - alpha) * _BASE_BUDGET[0] + alpha * _SHIFTED_BUDGET[0]
            hi = (1 - alpha) * _BASE_BUDGET[1] + alpha * _SHIFTED_BUDGET[1]
            bud.append((lo, hi))
        else:
            bud.append(_BASE_BUDGET)
    return acc, bud


def _sample_profile(rng: np.random.Generator, use_case: str,
                    acc_req_range: tuple[float, float] = _BASE_ACC_REQ,
                    budget_range: tuple[float, float] = _BASE_BUDGET) -> Profile:
    spec = _USE_CASE_SPECS[use_case]
    return Profile(
        use_case=use_case,
        monthly_queries=_log_uniform(rng, 1e5, 3e7),
        latency_sla_ms=_log_uniform(rng, *spec["sla"]),
        acc_req=float(rng.uniform(*acc_req_range)),
        monthly_budget_usd=_log_uniform(rng, *budget_range),
        data_sensitivity=SENSITIVITY[int(rng.integers(3))],
        peak_multiplier=float(rng.uniform(1.5, 5.0)),
        avg_input_tokens=_log_uniform(rng, *spec["inp"]),
        avg_output_tokens=_log_uniform(rng, *spec["out"]),
        concurrent_users=_log_uniform(rng, 10, 5_000),
    )


def generate_drift_stream(weeks: int = 4, per_week: int = 250,
                          pre_drift_weeks: int = 1, seed: int = 0,
                          drift_acc_req: bool = False,
                          drift_budget: bool = False) -> tuple[list[Profile], np.ndarray]:
    """Return (profiles, week_index per profile).

    Three independent drift knobs, all linearly interpolated from the end of
    pre_drift_weeks to the final week:
      - use-case mix: shifts toward chatbot + rag heavy (always on)
      - drift_acc_req: acc_req range Uniform(0.60, 0.90) -> Uniform(0.75, 0.95)
      - drift_budget:  monthly budget log-Uniform($1k, $30k) -> log-Uniform($500, $15k)
    """
    rng = np.random.default_rng(seed)
    mix = drift_mix(weeks, pre_drift_weeks)
    acc_ranges, bud_ranges = _week_ranges(weeks, pre_drift_weeks, drift_acc_req, drift_budget)
    profiles, week_idx = [], []
    for w in range(weeks):
        for _ in range(per_week):
            uc = USE_CASES[int(rng.choice(len(USE_CASES), p=mix[w]))]
            profiles.append(_sample_profile(rng, uc, acc_ranges[w], bud_ranges[w]))
            week_idx.append(w)
    return profiles, np.array(week_idx)


@dataclass
class DriftResult:
    rewards_static: np.ndarray
    rewards_online: np.ndarray
    week: np.ndarray

    def weekly(self, which: str = "online") -> np.ndarray:
        r = self.rewards_online if which == "online" else self.rewards_static
        return np.array([r[self.week == w].mean() for w in np.unique(self.week)])


def run_drift_experiment(agent_factory, pretrain_profiles: list[Profile],
                         weeks: int = 4, per_week: int = 250,
                         pre_drift_weeks: int = 1, seed: int = 0,
                         drift_acc_req: bool = False,
                         drift_budget: bool = False) -> DriftResult:
    """Train one agent on pretrain_profiles, then clone it: one copy runs frozen
    (static), the other keeps updating on the drift stream (online)."""
    from ..evaluation.experiments import train

    # pretrain to convergence on the stationary distribution
    pre_env = InfraEnv(pretrain_profiles, seed=seed)
    base_agent = agent_factory(seed)
    train(pre_env, base_agent, pretrain_profiles, epochs=20, seed=seed)

    stream, week_idx = generate_drift_stream(weeks, per_week, pre_drift_weeks,
                                             seed=seed + 100,
                                             drift_acc_req=drift_acc_req,
                                             drift_budget=drift_budget)
    static_agent = copy.deepcopy(base_agent)
    online_agent = copy.deepcopy(base_agent)

    env = InfraEnv(stream)
    rewards_s = np.zeros(len(stream))
    rewards_o = np.zeros(len(stream))
    for i, p in enumerate(stream):
        rewards_s[i] = run_episode(env, static_agent, p, train=False).reward
        rewards_o[i] = run_episode(env, online_agent, p, train=True).reward
    return DriftResult(rewards_static=rewards_s, rewards_online=rewards_o, week=week_idx)


def main() -> None:
    import argparse
    import csv
    import os

    from ..agents.bandit import ContextualBandit
    from ..evaluation.plots import plot_drift
    from ..simulator.profiles import generate_profiles

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weeks", type=int, default=4)
    ap.add_argument("--per-week", type=int, default=250)
    ap.add_argument("--pre-drift-weeks", type=int, default=1)
    ap.add_argument("--pretrain-n", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results/drift")
    ap.add_argument("--alpha", type=float, default=0.3, help="LinUCB exploration coefficient")
    ap.add_argument("--drift-acc-req", action="store_true",
                    help="also drift acc_req range from Uniform(0.60, 0.90) to Uniform(0.75, 0.95)")
    ap.add_argument("--drift-budget", action="store_true",
                    help="also drift monthly budget from log-U($1k, $30k) to log-U($500, $15k)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    all_rows = []
    all_results = []
    for seed in range(args.seeds):
        pretrain = generate_profiles(args.pretrain_n, seed=100 + seed)
        res = run_drift_experiment(
            agent_factory=lambda s, a=args.alpha: ContextualBandit(seed=s, alpha=a),
            pretrain_profiles=pretrain,
            weeks=args.weeks, per_week=args.per_week,
            pre_drift_weeks=args.pre_drift_weeks, seed=seed,
            drift_acc_req=args.drift_acc_req, drift_budget=args.drift_budget,
        )
        all_results.append(res)
        for i in range(len(res.week)):
            all_rows.append({
                "seed": seed, "step": i, "week": int(res.week[i]),
                "reward_static": float(res.rewards_static[i]),
                "reward_online": float(res.rewards_online[i]),
            })
        print(f"[drift seed {seed}] static weekly: {np.round(res.weekly('static'), 3).tolist()}  "
              f"online weekly: {np.round(res.weekly('online'), 3).tolist()}", flush=True)

    csv_path = f"{args.out}/drift.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "step", "week", "reward_static", "reward_online"])
        w.writeheader()
        w.writerows(all_rows)

    plot_drift(all_results, f"{args.out}/drift.png", pre_drift_weeks=args.pre_drift_weeks)

    static_mean = np.mean([r.rewards_static.mean() for r in all_results])
    online_mean = np.mean([r.rewards_online.mean() for r in all_results])
    print(f"\nOverall mean reward (over {args.seeds} seeds): static {static_mean:.3f}  online {online_mean:.3f}  "
          f"delta {online_mean - static_mean:+.3f}")


if __name__ == "__main__":
    main()
