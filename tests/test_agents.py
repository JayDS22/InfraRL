import numpy as np

from infrarl.agents.base import Agent
from infrarl.agents.bandit import ContextFreeBandit, ContextualBandit
from infrarl.agents.heuristic import HeuristicAgent
from infrarl.agents.random_agent import RandomAgent
from infrarl.evaluation.experiments import evaluate, run_episode, summarize
from infrarl.simulator.environment import InfraEnv
from infrarl.simulator.profiles import generate_profiles

ALL = [lambda: RandomAgent(0), HeuristicAgent, lambda: ContextFreeBandit(seed=0), lambda: ContextualBandit(seed=0)]


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
