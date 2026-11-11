"""Verification suite for the simulator, action masking, reward and determinism.

    python -m infrarl.evaluation.verify --out results/verification.json

Checks (each reports n_checked / n_failed):
  1. Mask enumeration: depth-first search over EVERY legal action sequence for 1- and
     2-tier stacks and a random sample of 3-tier stacks. Verifies no dead ends (a legal
     action always exists), the path count equals the closed form T^k * 3 (T = number of
     memory-feasible (hardware, model, precision) tiers), no terminal stack contains an
     infeasible tier, every action outside the mask is rejected by env.step, and the
     reward is exactly 0 before the terminal step.
  2. Simulator properties on random (profile, tier) pairs: quality rises with model
     size, latency falls with faster hardware / lower precision, quantisation never
     raises quality, cost is ordered static >= conservative >= aggressive, cost is
     non-decreasing in traffic.
  3. Reward bounds and boundary cases.
  4. Determinism: same seeds -> identical profiles, rollouts and agent results.
  5. Oracle gap: brute-force optimum over all single-tier stacks per profile.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json

import numpy as np

from ..agents.bandit import ContextualBandit
from ..agents.heuristic import HeuristicAgent
from ..simulator.environment import (
    N_ACTIONS, InfraEnv, episode_length, terminal_reward,
)
from ..simulator.estimator import SCALING_POLICIES, Stack, Tier, estimate, is_feasible
from ..simulator.hardware import HARDWARE, HARDWARE_NAMES
from ..simulator.models import MODEL_NAMES, PRECISION_NAMES, min_memory_gb
from ..simulator.profiles import Profile, generate_profiles
from .experiments import evaluate


def _one_profile_env() -> InfraEnv:
    p = Profile("chatbot", 2e6, 1000, 0.8, 5000, "low", 3.0, 500, 200, 200)
    return InfraEnv([p], seed=0)


def feasible_tier_count() -> int:
    return sum(is_feasible(Tier(h, m, p)) for h, m, p in itertools.product(HARDWARE_NAMES, MODEL_NAMES, PRECISION_NAMES))


def check_masks(n_three_tier_samples: int = 20000, seed: int = 0) -> dict:
    T = feasible_tier_count()
    # independent re-derivation of feasibility straight from the memory formula
    T_independent = sum(
        min_memory_gb(m, p) <= HARDWARE[h]["memory_gb"]
        for h in HARDWARE_NAMES for m in MODEL_NAMES for p in PRECISION_NAMES
    )
    res = {"feasible_tiers": T, "feasible_tiers_independent": T_independent,
           "nodes": 0, "dead_ends": 0, "masked_actions_accepted": 0, "nonzero_nonterminal_reward": 0,
           "infeasible_terminal_stacks": 0, "terminal_errors": 0, "bad_episode_length": 0}
    paths = {1: 0, 2: 0}

    def visit(env, depth, tiers_done):
        mask = env.action_masks()
        res["nodes"] += 1
        if not mask.any():
            res["dead_ends"] += 1
            return
        for a in range(N_ACTIONS):
            e = copy.deepcopy(env)
            if not mask[a]:
                try:
                    e.step(a)
                    res["masked_actions_accepted"] += 1
                except ValueError:
                    pass
                continue
            if depth == 0 and a - 0 + 1 > 2:  # exhaustive DFS only for 1- and 2-tier stacks
                continue
            try:
                _, r, term, _, info = e.step(a)
            except Exception:
                res["terminal_errors"] += 1
                continue
            if term:
                s = info["stack"]
                if not all(is_feasible(t) for t in s.tiers):
                    res["infeasible_terminal_stacks"] += 1
                if depth + 1 != episode_length(s.tier_count):
                    res["bad_episode_length"] += 1
                paths[s.tier_count] += 1
            else:
                if r != 0.0:
                    res["nonzero_nonterminal_reward"] += 1
                visit(e, depth + 1, tiers_done)

    env = _one_profile_env()
    env.reset(seed=0)
    visit(env, 0, 0)
    res["paths_1_tier"], res["paths_2_tier"] = paths[1], paths[2]
    res["expected_paths_1_tier"] = T * 3
    res["expected_paths_2_tier"] = T ** 2 * 3
    res["path_counts_match_closed_form"] = paths[1] == T * 3 and paths[2] == T ** 2 * 3

    # random sample of 3-tier episodes (full space has T^3 * 3 paths)
    rng = np.random.default_rng(seed)
    env = _one_profile_env()
    three = bad = 0
    for _ in range(n_three_tier_samples):
        env.reset()
        _, info = env.reset()
        env.step(2)  # three-tier architecture
        steps = 1
        while True:
            m = env.action_masks()
            if not m.any():
                res["dead_ends"] += 1
                break
            try:
                _, r, term, _, info = env.step(int(rng.choice(np.flatnonzero(m))))
            except Exception:
                res["terminal_errors"] += 1
                break
            steps += 1
            if term:
                three += 1
                bad += not all(is_feasible(t) for t in info["stack"].tiers)
                res["bad_episode_length"] += steps != episode_length(3)
                break
    res["three_tier_sampled"] = three
    res["infeasible_terminal_stacks"] += bad
    res["n_failed"] = (res["dead_ends"] + res["masked_actions_accepted"] + res["nonzero_nonterminal_reward"]
                       + res["infeasible_terminal_stacks"] + res["terminal_errors"] + res["bad_episode_length"]
                       + (0 if res["path_counts_match_closed_form"] and T == T_independent else 1))
    return res


def check_simulator(n_profiles: int = 300, seed: int = 0) -> dict:
    profiles = generate_profiles(n_profiles, seed)
    fails = {"quality_vs_model_size": 0, "latency_vs_hardware": 0, "latency_vs_precision": 0,
             "quantisation_never_raises_quality": 0, "cost_scaling_order": 0, "cost_vs_traffic": 0,
             "quality_independent_of_hardware": 0}
    checked = dict.fromkeys(fails, 0)
    speed_order = sorted(HARDWARE_NAMES, key=lambda h: HARDWARE[h]["tps_13b_fp16"])  # slow -> fast
    for p in profiles:
        # quality increases with model size at fixed hardware/precision
        for h, pr in itertools.product(HARDWARE_NAMES, PRECISION_NAMES):
            tiers = [Tier(h, m, pr) for m in MODEL_NAMES if is_feasible(Tier(h, m, pr))]
            qs = [estimate(Stack((t,), "static"), p)[0] for t in tiers]
            checked["quality_vs_model_size"] += 1
            fails["quality_vs_model_size"] += not all(a <= b for a, b in zip(qs, qs[1:]))
        # per-request latency decreases with faster hardware at fixed model/precision (static, ignoring queueing)
        for m, pr in itertools.product(MODEL_NAMES, PRECISION_NAMES):
            ts = [Tier(h, m, pr) for h in speed_order if is_feasible(Tier(h, m, pr))]
            from ..simulator.estimator import tokens_per_sec
            tps = [tokens_per_sec(t) for t in ts]
            checked["latency_vs_hardware"] += 1
            fails["latency_vs_hardware"] += not all(a < b for a, b in zip(tps, tps[1:]))
        # lower precision is faster and never higher quality
        for h, m in itertools.product(HARDWARE_NAMES, MODEL_NAMES):
            ts = [Tier(h, m, pr) for pr in PRECISION_NAMES if is_feasible(Tier(h, m, pr))]
            from ..simulator.estimator import tokens_per_sec, tier_quality
            sp = [tokens_per_sec(t) for t in ts]  # int4 .. fp32 : decreasing speed
            checked["latency_vs_precision"] += 1
            fails["latency_vs_precision"] += not all(a > b for a, b in zip(sp, sp[1:]))
            ql = [tier_quality(t, p.use_case) for t in ts]
            checked["quantisation_never_raises_quality"] += 1
            fails["quantisation_never_raises_quality"] += not all(a <= b for a, b in zip(ql, ql[1:]))
        # quality does not depend on hardware
        t = [Tier(h, "7b", "fp16") for h in HARDWARE_NAMES]
        qs = {round(estimate(Stack((x,), "static"), p)[0], 9) for x in t}
        checked["quality_independent_of_hardware"] += 1
        fails["quality_independent_of_hardware"] += len(qs) != 1
        # cost: static >= conservative >= aggressive; non-decreasing in traffic
        tier = Tier("a100", "13b", "fp16")
        costs = [estimate(Stack((tier,), s), p)[2] for s in SCALING_POLICIES]
        checked["cost_scaling_order"] += 1
        fails["cost_scaling_order"] += not (costs[0] >= costs[1] >= costs[2])
        heavier = Profile(**{**p.__dict__, "monthly_queries": p.monthly_queries * 4})
        checked["cost_vs_traffic"] += 1
        fails["cost_vs_traffic"] += estimate(Stack((tier,), "static"), heavier)[2] < estimate(Stack((tier,), "static"), p)[2]
    return {"checked": checked, "failed": fails, "n_failed": sum(fails.values())}


def check_reward() -> dict:
    p = Profile("chatbot", 1e6, 500, 0.8, 1000, "low", 2.0, 100, 100, 10)
    cases = [
        # (quality, latency, cost) -> expected reward
        ((0.8, 500, 1000), 0.5 * 1.0 + 0.3 * 1.0 + 0.2 * 1.0),            # all boundaries inclusive
        ((0.8, 500, 1000.01), 0.5 + 0.0 + 0.2),                            # just over budget: B_r = 0
        ((0.8, 500.01, 1000), 0.5 + 0.3 + 0.0),                            # just over SLA: S_r = 0
        ((0.4, 500, 1000), 0.25 + 0.3 + 0.2),                              # half the required quality
        ((1.6, 0.0, 0.0), 0.5 + 0.3 * 1.1 + 0.2),                          # quality clipped, full savings bonus
        ((0.0, 1e9, 1e9), 0.0),                                           # worst case
    ]
    fails = 0
    for (q, l, c), expected in cases:
        r, _ = terminal_reward(q, l, c, p)
        fails += abs(r - expected) > 1e-9
    return {"checked": len(cases), "n_failed": int(fails), "max_reward": 1.03}


def check_determinism(seed: int = 0) -> dict:
    a, b = generate_profiles(200, seed), generate_profiles(200, seed)
    ok_profiles = a == b
    outs = []
    for _ in range(2):
        env = InfraEnv(a, seed=seed)
        out = evaluate(env, ContextualBandit(seed=seed), a[:100], train_epochs=3, seed=seed)
        outs.append([(round(r.reward, 9), r.stack) for r in out])
    h = [[(round(r.reward, 9), r.stack) for r in evaluate(InfraEnv(a), HeuristicAgent(), a[:100])] for _ in range(2)]
    ok = ok_profiles and outs[0] == outs[1] and h[0] == h[1]
    return {"checked": 3, "n_failed": int(not ok)}


def oracle_gap(n: int = 1000, seed: int = 1000) -> dict:
    """Brute-force best single-tier stack per profile vs the heuristic and the context-aware bandit."""
    profiles = generate_profiles(n, seed)
    stacks = [Stack((Tier(h, m, pr),), s)
              for h, m, pr, s in itertools.product(HARDWARE_NAMES, MODEL_NAMES, PRECISION_NAMES, SCALING_POLICIES)
              if is_feasible(Tier(h, m, pr))]
    oracle, satisfiable = [], 0
    for p in profiles:
        best = 0.0
        sat = False
        for s in stacks:
            q, l, c = estimate(s, p)
            r, parts = terminal_reward(q, l, c, p)
            best = max(best, r)
            sat |= parts["within_budget"] and parts["meets_sla"] and q >= p.acc_req
        oracle.append(best)
        satisfiable += sat
    env = InfraEnv(profiles)
    agents = {"heuristic": HeuristicAgent(), "bandit_contextual": ContextualBandit(seed=0)}
    train_profiles = generate_profiles(1000, 2000)
    out = {"n_profiles": n, "n_single_tier_stacks": len(stacks), "oracle_mean_reward": float(np.mean(oracle)),
           "fully_satisfiable_fraction": satisfiable / n}
    for name, agent in agents.items():
        res = evaluate(InfraEnv(train_profiles), agent, profiles, train_epochs=20 if name != "heuristic" else 0, seed=0)
        gap = np.array(oracle) - np.array([r.reward for r in res])
        out[name] = {"mean_reward": float(np.mean([r.reward for r in res])), "mean_gap": float(gap.mean()),
                     "within_0.01_of_oracle": float((gap <= 0.01).mean()), "above_oracle": float((gap < -1e-9).mean())}
    return out


def check_price_anchor() -> dict:
    """Published Nebius on-demand price (nebius.com/prices, accessed 2026-10-25): HGX H100 = $4.50 / GPU-hour.
    No A100 or A10 on that page, so those two prices have no Nebius anchor."""
    published = {"h100": 4.50}
    return {hw: {"published": v, "simulator": HARDWARE[hw]["cost_per_hr_usd"],
                 "residual_pct": 100 * abs(HARDWARE[hw]["cost_per_hr_usd"] - v) / v} for hw, v in published.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/verification.json")
    args = ap.parse_args()
    out = {
        "masks": check_masks(),
        "simulator": check_simulator(),
        "reward": check_reward(),
        "determinism": check_determinism(),
        "oracle": oracle_gap(),
        "price_anchor": check_price_anchor(),
    }
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    failed = out["masks"]["n_failed"] + out["simulator"]["n_failed"] + out["reward"]["n_failed"] + out["determinism"]["n_failed"]
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
