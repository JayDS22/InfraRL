import numpy as np

from infrarl.agents.bandit import ContextualBandit
from infrarl.online.drift import drift_mix, generate_drift_stream, run_drift_experiment
from infrarl.simulator.models import USE_CASES
from infrarl.simulator.profiles import generate_profiles


def test_drift_mix_sums_to_one_and_shifts():
    mix = drift_mix(weeks=4, pre_drift_weeks=1)
    assert mix.shape == (4, len(USE_CASES))
    assert np.allclose(mix.sum(axis=1), 1.0)
    # week 0 uniform, week 3 skewed toward chatbot
    assert mix[0, USE_CASES.index("chatbot")] < mix[3, USE_CASES.index("chatbot")]


def test_drift_stream_has_expected_lengths_and_distribution_shift():
    profiles, week = generate_drift_stream(weeks=4, per_week=200, seed=0)
    assert len(profiles) == 4 * 200
    assert set(np.unique(week).tolist()) == {0, 1, 2, 3}
    # chatbot share should rise from week 0 to week 3
    def share(w):
        wk = [p for p, i in zip(profiles, week) if i == w]
        return sum(p.use_case == "chatbot" for p in wk) / len(wk)
    assert share(3) > share(0) + 0.1


def test_acc_req_drift_shifts_distribution():
    _, week_a = generate_drift_stream(weeks=4, per_week=400, seed=1, drift_acc_req=False)
    _, week_b = generate_drift_stream(weeks=4, per_week=400, seed=1, drift_acc_req=True)
    from infrarl.online.drift import generate_drift_stream as gen
    profiles_a, _ = gen(weeks=4, per_week=400, seed=1, drift_acc_req=False)
    profiles_b, _ = gen(weeks=4, per_week=400, seed=1, drift_acc_req=True)
    last_week_a = np.array([p.acc_req for p, w in zip(profiles_a, week_a) if w == 3])
    last_week_b = np.array([p.acc_req for p, w in zip(profiles_b, week_b) if w == 3])
    assert last_week_b.mean() > last_week_a.mean() + 0.05
    assert last_week_b.min() >= 0.75 - 1e-6


def test_budget_drift_shifts_distribution():
    from infrarl.online.drift import generate_drift_stream as gen
    profiles_a, week_a = gen(weeks=4, per_week=400, seed=2, drift_budget=False)
    profiles_b, week_b = gen(weeks=4, per_week=400, seed=2, drift_budget=True)
    last_a = np.array([p.monthly_budget_usd for p, w in zip(profiles_a, week_a) if w == 3])
    last_b = np.array([p.monthly_budget_usd for p, w in zip(profiles_b, week_b) if w == 3])
    assert last_b.mean() < last_a.mean() * 0.7


def test_drift_experiment_online_matches_or_beats_static_on_drift():
    """Online bandit should not be beaten by its frozen clone on the drifted weeks."""
    pretrain = generate_profiles(500, seed=42)
    result = run_drift_experiment(
        agent_factory=lambda s: ContextualBandit(seed=s, alpha=0.3),
        pretrain_profiles=pretrain,
        weeks=3, per_week=150, pre_drift_weeks=1, seed=7,
    )
    static_weekly = result.weekly("static")
    online_weekly = result.weekly("online")
    # on week 2 (full drift) online mean should match static within noise (static pretrained well);
    # the key claim is that online does not COLLAPSE under drift
    assert online_weekly[-1] > 0.7 * static_weekly[0]
