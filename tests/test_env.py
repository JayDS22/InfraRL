import numpy as np

from infrarl.simulator.environment import (
    N_ACTIONS, OBS_DIM, InfraEnv, episode_length, terminal_reward,
)
from infrarl.simulator.profiles import generate_profiles


def _rollout(env, rng):
    obs, info = env.reset()
    steps, total = 0, 0.0
    while True:
        mask = info["action_mask"]
        assert mask.any()
        obs, r, term, trunc, info = env.step(int(rng.choice(np.flatnonzero(mask))))
        assert obs.shape == (OBS_DIM,)
        steps += 1
        total += r
        if term:
            return steps, total, info


def test_episode_length_matches_tier_count():
    env = InfraEnv(generate_profiles(50, 0), seed=0)
    rng = np.random.default_rng(0)
    for _ in range(200):
        steps, _, info = _rollout(env, rng)
        assert steps == episode_length(info["stack"].tier_count)


def test_reward_is_terminal_only_and_bounded():
    env = InfraEnv(generate_profiles(50, 0), seed=1)
    rng = np.random.default_rng(1)
    for _ in range(100):
        obs, info = env.reset()
        while True:
            obs, r, term, _, info = env.step(int(rng.choice(np.flatnonzero(info["action_mask"]))))
            if not term:
                assert r == 0.0
            else:
                assert 0.0 <= r <= 1.03 + 1e-9
                break


def test_masks_never_allow_infeasible_hardware_combo():
    env = InfraEnv(generate_profiles(50, 0), seed=2)
    rng = np.random.default_rng(2)
    for _ in range(300):
        _, _, info = _rollout(env, rng)  # estimate() raises on infeasible tiers
        assert info["quality"] > 0


def test_masked_action_rejected_and_done_state():
    env = InfraEnv(generate_profiles(5, 0))
    env.reset()
    try:
        env.step(5)  # hardware action at the architecture step
    except ValueError:
        pass
    else:
        raise AssertionError("masked action accepted")


def test_action_space_size():
    assert InfraEnv(generate_profiles(2, 0)).action_space.n == N_ACTIONS == 18


def test_terminal_reward_formula():
    p = generate_profiles(1, 0)[0]
    r, parts = terminal_reward(p.acc_req, p.latency_sla_ms, p.monthly_budget_usd, p)
    assert abs(r - (0.5 + 0.3 + 0.2)) < 1e-9
    r2, _ = terminal_reward(p.acc_req * 2, 0.0, 0.0, p)
    assert abs(r2 - 1.03) < 1e-9
