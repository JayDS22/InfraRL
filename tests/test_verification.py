"""The verification suite must pass on correct code AND fail when a bug is injected."""
import infrarl.simulator.environment as envmod
from infrarl.evaluation import verify


def test_verification_passes_on_correct_code():
    assert verify.check_masks(n_three_tier_samples=300)["n_failed"] == 0
    assert verify.check_simulator(n_profiles=20)["n_failed"] == 0
    assert verify.check_reward()["n_failed"] == 0
    assert verify.check_determinism()["n_failed"] == 0


def test_injected_mask_bug_is_caught(monkeypatch):
    # bug: memory feasibility ignored when building masks -> infeasible stacks become reachable
    monkeypatch.setattr(envmod, "min_memory_gb", lambda model, precision: 0.0)
    out = verify.check_masks(n_three_tier_samples=50)
    assert out["n_failed"] > 0


def test_injected_simulator_bug_is_caught(monkeypatch):
    # bug: int4 given a higher quality than fp16
    import infrarl.simulator.estimator as est
    monkeypatch.setitem(est.PRECISION_QUALITY_PENALTY, "int4", -0.2)
    assert verify.check_simulator(n_profiles=5)["n_failed"] > 0


def test_price_anchor_h100_matches_published():
    assert verify.check_price_anchor()["h100"]["residual_pct"] <= 15.0
