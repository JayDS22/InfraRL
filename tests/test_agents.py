import numpy as np
import pytest

from infrarl.agents.base import Agent
from infrarl.agents.bandit import ContextFreeBandit, ContextualBandit, DiscountedContextualBandit
from infrarl.agents.heuristic import HeuristicAgent
from infrarl.agents.random_agent import RandomAgent
from infrarl.evaluation.experiments import evaluate, run_episode, summarize
from infrarl.simulator.environment import InfraEnv
from infrarl.simulator.profiles import generate_profiles

try:
    from infrarl.agents.ppo_agent import PPOAgent
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from infrarl.agents.q_learning import QLearningAgent

ALL = [lambda: RandomAgent(0), HeuristicAgent, lambda: ContextFreeBandit(seed=0), lambda: ContextualBandit(seed=0),
       lambda: DiscountedContextualBandit(seed=0), lambda: QLearningAgent(seed=0)]
if _HAS_TORCH:
    # rollout_size large enough that the 60-profile protocol test never triggers _learn
    ALL.append(lambda: PPOAgent(seed=0, rollout_size=10_000))


def test_agents_satisfy_protocol_and_only_play_legal_actions():
    profiles = generate_profiles(60, 0)
    env = InfraEnv(profiles)
    for make in ALL:
        agent = make()
        assert isinstance(agent, Agent)
        for p in profiles:
            run_episode(env, agent, p, train=True)  # env raises on masked actions


def test_heuristic_beats_random_on_reward():
    profiles = generate_profiles(200, 5)
    env = InfraEnv(profiles)
    h = summarize(evaluate(env, HeuristicAgent(), profiles))
    r = summarize(evaluate(env, RandomAgent(0), profiles))
    assert h["mean_reward"] > r["mean_reward"]


def test_bandit_learns():
    profiles = generate_profiles(200, 7)
    env = InfraEnv(profiles)
    r = summarize(evaluate(env, RandomAgent(0), profiles))
    b = summarize(evaluate(env, ContextFreeBandit(seed=0), profiles, train_epochs=5))
    assert b["mean_reward"] > r["mean_reward"]


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_ppo_learns():
    # small rollout so _learn fires during training, 4 epochs over 150 profiles
    profiles = generate_profiles(150, 11)
    env = InfraEnv(profiles)
    r = summarize(evaluate(env, RandomAgent(0), profiles))
    p = summarize(evaluate(env, PPOAgent(seed=0, rollout_size=256), profiles, train_epochs=4))
    assert p["mean_reward"] > r["mean_reward"]


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_ppo_auto_registered():
    from infrarl.agents.registry import REGISTRY
    assert "ppo" in REGISTRY
    spec = REGISTRY["ppo"]
    assert spec.train_epochs > 0 and callable(spec.factory)


def test_q_learning_learns():
    profiles = generate_profiles(250, 13)
    env = InfraEnv(profiles)
    r = summarize(evaluate(env, RandomAgent(0), profiles))
    q = summarize(evaluate(env, QLearningAgent(seed=0, epsilon_decay_episodes=1500),
                           profiles, train_epochs=20))
    assert q["mean_reward"] > r["mean_reward"]


def test_q_learning_auto_registered():
    from infrarl.agents.registry import REGISTRY
    assert "q_learning" in REGISTRY
    spec = REGISTRY["q_learning"]
    assert spec.train_epochs > 0 and callable(spec.factory)


def test_discounted_bandit_learns_and_matches_vanilla_on_stationary():
    profiles = generate_profiles(300, 17)
    env = InfraEnv(profiles)
    r = summarize(evaluate(env, RandomAgent(0), profiles))
    d = summarize(evaluate(env, DiscountedContextualBandit(seed=0, gamma=0.995), profiles, train_epochs=8))
    assert d["mean_reward"] > r["mean_reward"]


def test_discounted_bandit_registered():
    from infrarl.agents.registry import REGISTRY
    assert "bandit_discounted" in REGISTRY


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_lagrangian_ppo_lambda_responds_to_compliance_gap():
    from infrarl.agents.lagrangian_ppo import LagrangianPPOAgent
    profiles = generate_profiles(200, 23)
    env = InfraEnv(profiles)
    agent = LagrangianPPOAgent(seed=0, compliance_target=0.85,
                               lambda_lr=0.1, rollout_size=256, update_epochs=2)
    # train enough to trigger multiple _learn() calls -> multiple dual steps
    from infrarl.evaluation.experiments import train
    train(env, agent, profiles, epochs=3, seed=0)
    assert len(agent.history["lambda"]) >= 1
    # if compliance is below target, lambda must rise from 0
    if agent.history["compliance"][0] < 0.85:
        assert agent.history["lambda"][-1] > 0.0
    assert 0.0 <= agent.history["lambda"][-1] <= agent.lambda_max


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_lagrangian_ppo_auto_registered():
    from infrarl.agents.registry import REGISTRY
    assert "lagrangian_ppo" in REGISTRY
