"""Lagrangian PPO for the quality-compliance-constrained InfraRL CMDP.

Constrained MDP formulation (Altman 1999): maximize E[sum r_t] subject to
E[compliance] >= target, where compliance = I[q_T >= acc_req]. Rather than
scalarize via reward weights (the vanilla PPO path), we introduce a Lagrange
multiplier lambda >= 0 and alternate:

    primal: maximize L(pi, lambda) = E[sum r_t - lambda * (target - compliance)]
    dual:   lambda_{t+1} = max(0, lambda_t + eta_lam * (target - mean_compliance))

The primal step is a standard PPO update on a reshaped reward
r' = r - lambda * (1 - compliant) applied at terminal transitions (the
constraint is terminal-only in our MDP since compliance is only defined at
episode end). The dual step is projected gradient ascent on lambda with a
small learning rate; we clip lambda to [0, lambda_max] for stability.

Vanilla dual ascent can oscillate near the constraint (Stooke, Achiam, Abbeel
2020 "Responsive Safety in Reinforcement Learning by PID Lagrangian Methods");
we expose an optional integral-and-derivative term that falls back to vanilla
Lagrangian when PID gains are 0 (the default).

Primary citations:
  Altman (1999) Constrained Markov Decision Processes.
  Achiam, Held, Tamar, Abbeel (ICML 2017) Constrained Policy Optimization.
  Stooke, Achiam, Abbeel (ICML 2020) PID Lagrangian methods.
"""
from __future__ import annotations

import numpy as np

from .ppo_agent import PPOAgent


class LagrangianPPOAgent(PPOAgent):
    name = "lagrangian_ppo"

    def __init__(
        self,
        seed: int = 0,
        compliance_target: float = 0.85,
        lambda_init: float = 0.0,
        lambda_lr: float = 0.03,
        lambda_max: float = 5.0,
        # PID knobs (Stooke 2020). Default integral/derivative gains = 0 reduces
        # to vanilla dual ascent; enable by setting ki > 0 or kd > 0.
        ki: float = 0.0,
        kd: float = 0.0,
        **ppo_kwargs,
    ):
        super().__init__(seed=seed, **ppo_kwargs)
        self.compliance_target = compliance_target
        self.lambda_ = float(lambda_init)
        self.lambda_lr = lambda_lr
        self.lambda_max = lambda_max
        self.ki = ki
        self.kd = kd
        self._err_integral = 0.0
        self._err_prev = 0.0
        # Diagnostics tracked across _learn calls; inspected by tests + analysis.
        self.history = {"lambda": [], "compliance": [], "mean_reward": []}

    def _reshape_rewards(self, rews: np.ndarray, dones: np.ndarray,
                         compliant: np.ndarray) -> np.ndarray:
        # Penalize non-compliant terminals; non-terminal rewards are 0 by env
        # convention so (1 - compliant) is multiplied by the terminal indicator
        # to avoid penalizing intermediate steps.
        penalty = self.lambda_ * dones.astype(np.float32) * (1.0 - compliant)
        return rews - penalty

    def _dual_step(self, mean_compliance: float) -> None:
        error = self.compliance_target - mean_compliance
        self._err_integral += error
        d_err = error - self._err_prev
        self._err_prev = error
        step = self.lambda_lr * error + self.ki * self._err_integral + self.kd * d_err
        self.lambda_ = float(np.clip(self.lambda_ + step, 0.0, self.lambda_max))

    def _learn(self) -> None:
        # Capture terminals' compliance BEFORE the primal PPO update so the dual
        # step reflects the policy that produced this rollout.
        dones = np.array([x[6] for x in self._buf], dtype=bool)
        compliant = np.array([x[7] for x in self._buf], dtype=np.float32)
        terminal_compliance = compliant[dones]
        mean_compliance = float(terminal_compliance.mean()) if len(terminal_compliance) else 0.0

        # Primal PPO update using the reshaped reward (via _reshape_rewards hook).
        super()._learn()

        # Dual ascent on the Lagrange multiplier.
        self._dual_step(mean_compliance)

        # Diagnostics.
        self.history["lambda"].append(self.lambda_)
        self.history["compliance"].append(mean_compliance)
        rews = np.array([x[5] for x in self._buf], dtype=np.float32)
        self.history["mean_reward"].append(float(rews[dones].mean()) if dones.any() else 0.0)
