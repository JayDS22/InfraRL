import numpy as np

from infrarl.simulator.estimator import Stack, Tier, estimate, is_feasible
from infrarl.simulator.profiles import Profile, generate_profiles
from infrarl.simulator.models import USE_CASES


def _chatbot():
    return Profile("chatbot", 1e5, 500, 0.80, 2000, "low", 1.0, 100, 200, 50)


def test_matches_demo_smoke_test():
    q, l, c = estimate(Stack((Tier("a100", "13b", "fp16"),), "static"), _chatbot())
    assert (round(q, 2), round(l), round(c)) == (0.85, 400, 1728)


def test_memory_feasibility():
    assert not is_feasible(Tier("a10", "70b", "fp16"))
    assert not is_feasible(Tier("a10", "13b", "fp16"))
    assert is_feasible(Tier("a100", "70b", "int4"))
    assert is_feasible(Tier("cpu", "3b", "int4"))


def test_infeasible_stack_raises():
    try:
        estimate(Stack((Tier("a10", "70b", "fp16"),), "static"), _chatbot())
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_profiles_deterministic_and_cover_use_cases():
    a, b = generate_profiles(1000, seed=3), generate_profiles(1000, seed=3)
    assert a == b
    assert {p.use_case for p in a} == set(USE_CASES)
