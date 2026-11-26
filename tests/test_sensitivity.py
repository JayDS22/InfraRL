"""Unit tests for the simulator-perturbation harness.

Covers the two correctness properties that keep the sensitivity story honest:
1) The context manager restores env_module.estimate on exit (including on raise).
2) The nominal perturbation is a no-op; a non-identity perturbation changes
   the observed (quality, latency, cost) triple by the expected factor.
3) The grid builder emits exactly one nominal row regardless of axis list.
"""
import unittest

from infrarl.evaluation.sensitivity import (
    Perturbation,
    perturbation_grid,
    perturbed_estimate,
)
from infrarl.simulator import environment as env_module
from infrarl.simulator.estimator import Stack, Tier
from infrarl.simulator.profiles import generate_profiles


class SensitivityTests(unittest.TestCase):
    def setUp(self):
        self.profile = generate_profiles(1, seed=0)[0]
        self.stack = Stack((Tier("a100", "13b", "int8"),), "static")

    def test_context_restores_binding(self):
        before = env_module.estimate
        with perturbed_estimate(Perturbation(cost_mult=2.0)):
            self.assertIsNot(env_module.estimate, before)
        self.assertIs(env_module.estimate, before)

    def test_context_restores_on_exception(self):
        before = env_module.estimate
        with self.assertRaises(RuntimeError):
            with perturbed_estimate(Perturbation(cost_mult=2.0)):
                raise RuntimeError("boom")
        self.assertIs(env_module.estimate, before)

    def test_nominal_is_identity(self):
        q0, l0, c0 = env_module.estimate(self.stack, self.profile)
        with perturbed_estimate(Perturbation()):
            q1, l1, c1 = env_module.estimate(self.stack, self.profile)
        self.assertAlmostEqual(q0, q1)
        self.assertAlmostEqual(l0, l1)
        self.assertAlmostEqual(c0, c1)

    def test_multiplicative_applied(self):
        q0, l0, c0 = env_module.estimate(self.stack, self.profile)
        with perturbed_estimate(Perturbation(latency_mult=1.5, cost_mult=2.0)):
            q1, l1, c1 = env_module.estimate(self.stack, self.profile)
        self.assertAlmostEqual(q1, q0)
        self.assertAlmostEqual(l1, l0 * 1.5)
        self.assertAlmostEqual(c1, c0 * 2.0)

    def test_quality_clamped_to_unit_interval(self):
        with perturbed_estimate(Perturbation(quality_mult=100.0)):
            q, _, _ = env_module.estimate(self.stack, self.profile)
        self.assertLessEqual(q, 1.0)
        with perturbed_estimate(Perturbation(quality_mult=-1.0)):
            q, _, _ = env_module.estimate(self.stack, self.profile)
        self.assertGreaterEqual(q, 0.0)

    def test_grid_has_one_nominal(self):
        grid = perturbation_grid(("quality", "latency", "cost"), (0.8, 1.0, 1.2))
        nominals = [p for p in grid if p.is_nominal]
        self.assertEqual(len(nominals), 1)
        self.assertEqual(len(grid), 1 + 3 * 2)

    def test_grid_rejects_unknown_axis(self):
        with self.assertRaises(ValueError):
            perturbation_grid(("qulity",), (0.8,))


if __name__ == "__main__":
    unittest.main()
