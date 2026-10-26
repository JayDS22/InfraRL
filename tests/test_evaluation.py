import numpy as np
import pandas as pd

from infrarl.agents.bandit import ContextFreeBandit
from infrarl.agents.registry import REGISTRY, register
from infrarl.evaluation.ablations import ABLATIONS, AblatedInfraEnv
from infrarl.evaluation.experiments import evaluate, run_episode, summarize, train
from infrarl.evaluation.plots import pareto_frontier
from infrarl.evaluation.stats import compare_agents, holm
from infrarl.simulator.environment import InfraEnv
from infrarl.simulator.profiles import generate_profiles


def test_holm_is_monotone_and_bounded():
    adj = holm([0.01, 0.04, 0.03])
    assert adj[0] == 0.03 and max(adj) <= 1.0
    assert all(a >= p for a, p in zip(adj, [0.01, 0.04, 0.03]))


def test_compare_agents_detects_difference():
    rng = np.random.default_rng(0)
    rows = []
    for agent, shift in (("a", 0.0), ("b", 0.0), ("c", 0.5)):
        for i in range(100):
            rows.append(dict(agent=agent, seed=0, profile_idx=i, reward=rng.normal(shift, 0.1)))
    out = compare_agents(pd.DataFrame(rows), "reward")
    assert out[out.test == "friedman"].p_raw.iloc[0] < 1e-6
    ac = out[(out.a == "a") & (out.b == "c")].iloc[0]
    assert ac.p_holm < 0.01 and ac.mean_diff < 0


def test_pareto_frontier():
    assert pareto_frontier([(1, 0.5), (2, 0.9), (3, 0.6), (1.5, 0.4)]) == [0, 1]


def test_ablations_restrict_masks_and_arms():
    profiles = generate_profiles(30, 0)
    rng = np.random.default_rng(0)
    for ab in ABLATIONS.values():
        env = AblatedInfraEnv(ab, profiles)
        for _ in range(60):
            obs, info = env.reset()
            while True:
                mask = info["action_mask"]
                assert mask.any()
                obs, r, term, _, info = env.step(int(rng.choice(np.flatnonzero(mask))))
                if term:
                    s = info["stack"]
                    if ab.fixed_hardware:
                        assert all(t.hardware == ab.fixed_hardware for t in s.tiers)
                    if ab.static_scaling:
                        assert s.scaling == "static"
                    if ab.single_tier:
                        assert s.tier_count == 1
                    if ab.fp16_only:
                        assert all(t.precision in ("fp16", "int8", "int4") for t in s.tiers)
                    break


def test_ablated_bandit_never_plays_masked_action():
    profiles = generate_profiles(40, 1)
    for ab in ABLATIONS.values():
        env = AblatedInfraEnv(ab, profiles)
        agent = ContextFreeBandit(seed=0)
        agent.set_arm_filter(ab.arm_ok)
        assert agent.arm_allowed.any()
        for p in profiles:
            run_episode(env, agent, p, train=True)  # env raises on masked actions


def test_train_returns_curve_and_registry_plugin():
    profiles = generate_profiles(30, 2)
    env = InfraEnv(profiles)
    register("tmp_bandit", lambda s: ContextFreeBandit(seed=s), train_epochs=2)
    try:
        agent = REGISTRY["tmp_bandit"].factory(0)
        curve = train(env, agent, profiles, 2, eval_profiles=profiles)
        assert [c["epoch"] for c in curve] == [1, 2] and "eval_reward" in curve[0]
        s = summarize(evaluate(env, agent, profiles))
        assert 0 <= s["budget_compliance"] <= 1 and s["meets_all"] <= s["budget_compliance"]
    finally:
        REGISTRY.pop("tmp_bandit")
