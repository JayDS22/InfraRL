"""Gymnasium environment: 5-step (up to 11-step) masked-action MDP.

Flat Discrete(18) action space, partitioned into five groups:

    0-2    architecture   single-tier, two-tier, three-tier
    3-6    hardware       cpu, a10, a100, h100
    7-10   model          3b, 7b, 13b, 70b
    11-14  precision      int4, int8, fp16, fp32
    15-17  scaling        static, conservative, aggressive

Episode: arch, then (hardware, model, precision) per tier, then scaling.
Length = 3 * tier_count + 2. Reward is terminal only:

    r_T = 0.5 * Q_r + 0.3 * B_r + 0.2 * S_r
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .estimator import SCALING_POLICIES, Stack, Tier, estimate
from .hardware import HARDWARE, HARDWARE_NAMES
from .models import MODEL_NAMES, PRECISION_NAMES, USE_CASES, min_memory_gb
from .profiles import SENSITIVITY, Profile, generate_profiles

ARCH_RANGE = range(0, 3)
HW_RANGE = range(3, 7)
MODEL_RANGE = range(7, 11)
PREC_RANGE = range(11, 15)
SCALE_RANGE = range(15, 18)
N_ACTIONS = 18
MAX_TIERS = 3

STEP_ARCH, STEP_HW, STEP_MODEL, STEP_PREC, STEP_SCALE = range(5)
_STEP_RANGE = {
    STEP_ARCH: ARCH_RANGE,
    STEP_HW: HW_RANGE,
    STEP_MODEL: MODEL_RANGE,
    STEP_PREC: PREC_RANGE,
    STEP_SCALE: SCALE_RANGE,
}

REWARD_WEIGHTS = (0.5, 0.3, 0.2)  # quality, budget, SLA

N_PROFILE_FEATURES = 8 + len(USE_CASES) + len(SENSITIVITY)
# step type (5) + tier position (3) + arch (3) + per-tier hw/model/prec one-hots
OBS_DIM = N_PROFILE_FEATURES + 5 + MAX_TIERS + 3 + MAX_TIERS * 12


def episode_length(tier_count: int) -> int:
    return 3 * tier_count + 2


def terminal_reward(q: float, latency_ms: float, cost: float, profile: Profile) -> tuple[float, dict]:
    q_r = float(np.clip(q / profile.acc_req, 0.0, 1.0))
    b = profile.monthly_budget_usd
    within_budget = cost <= b
    b_r = float(within_budget) + 0.1 * max(0.0, (b - cost) / b)
    meets_sla = latency_ms <= profile.latency_sla_ms
    s_r = float(meets_sla)
    wq, wb, ws = REWARD_WEIGHTS
    reward = wq * q_r + wb * b_r + ws * s_r
    return reward, {"q_r": q_r, "b_r": b_r, "s_r": s_r, "within_budget": within_budget, "meets_sla": meets_sla}


def encode_profile(p: Profile) -> np.ndarray:
    numeric = [
        np.log10(p.monthly_queries) / 8.0,
        np.log10(p.latency_sla_ms) / 5.0,
        p.acc_req,
        np.log10(p.monthly_budget_usd) / 5.0,
        p.peak_multiplier / 5.0,
        np.log10(p.avg_input_tokens) / 4.0,
        np.log10(p.avg_output_tokens) / 4.0,
        np.log10(p.concurrent_users) / 4.0,
    ]
    uc = np.zeros(len(USE_CASES))
    uc[USE_CASES.index(p.use_case)] = 1.0
    sens = np.zeros(len(SENSITIVITY))
    sens[SENSITIVITY.index(p.data_sensitivity)] = 1.0
    return np.concatenate([numeric, uc, sens]).astype(np.float32)


class InfraEnv(gym.Env):
    """Sequential stack-construction environment with per-step action masks."""

    metadata = {"render_modes": []}

    def __init__(self, profiles: list[Profile] | None = None, seed: int = 0):
        super().__init__()
        self.profiles = profiles if profiles is not None else generate_profiles(1000, seed)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(0.0, 1.5, shape=(OBS_DIM,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self.profile: Profile | None = None
        self._profile_vec = np.zeros(N_PROFILE_FEATURES, dtype=np.float32)
        self._reset_episode_state()

    # ---- episode state -------------------------------------------------
    def _reset_episode_state(self) -> None:
        self.tier_count = 0
        self.tiers: list[list[str | None]] = []
        self.scaling: str | None = None
        self._schedule: list[tuple[int, int]] = [(STEP_ARCH, 0)]  # (step type, tier idx)
        self._t = 0
        self._done = False

    def _build_schedule(self, tier_count: int) -> None:
        sched = [(STEP_ARCH, 0)]
        for i in range(tier_count):
            sched += [(STEP_HW, i), (STEP_MODEL, i), (STEP_PREC, i)]
        sched.append((STEP_SCALE, 0))
        self._schedule = sched

    # ---- gym API -------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_episode_state()
        if options and "profile" in options:
            self.profile = options["profile"]
        else:
            self.profile = self.profiles[int(self._rng.integers(len(self.profiles)))]
        self._profile_vec = encode_profile(self.profile)
        return self._obs(), {"action_mask": self.action_masks(), "profile": self.profile}

    def step(self, action: int):
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset()")
        action = int(action)
        mask = self.action_masks()
        if not mask[action]:
            raise ValueError(f"action {action} is masked at step {self._t}")
        step_type, tier_idx = self._schedule[self._t]
        if step_type == STEP_ARCH:
            self.tier_count = action - ARCH_RANGE.start + 1
            self.tiers = [[None, None, None] for _ in range(self.tier_count)]
            self._build_schedule(self.tier_count)
        elif step_type == STEP_HW:
            self.tiers[tier_idx][0] = HARDWARE_NAMES[action - HW_RANGE.start]
        elif step_type == STEP_MODEL:
            self.tiers[tier_idx][1] = MODEL_NAMES[action - MODEL_RANGE.start]
        elif step_type == STEP_PREC:
            self.tiers[tier_idx][2] = PRECISION_NAMES[action - PREC_RANGE.start]
        else:
            self.scaling = SCALING_POLICIES[action - SCALE_RANGE.start]
        self._t += 1

        terminated = self._t >= len(self._schedule)
        reward = 0.0
        info: dict = {}
        if terminated:
            self._done = True
            stack = self.stack()
            q, lat, cost = estimate(stack, self.profile)
            reward, parts = terminal_reward(q, lat, cost, self.profile)
            info = {"stack": stack, "quality": q, "latency_ms": lat, "cost_usd": cost, **parts}
        info["action_mask"] = self.action_masks()
        return self._obs(), reward, terminated, False, info

    # ---- helpers -------------------------------------------------------
    def stack(self) -> Stack:
        return Stack(tuple(Tier(*t) for t in self.tiers), self.scaling)

    def current_step_type(self) -> int | None:
        return None if self._done else self._schedule[self._t][0]

    def action_masks(self) -> np.ndarray:
        """Boolean mask over the 18 actions for the current step."""
        mask = np.zeros(N_ACTIONS, dtype=bool)
        if self._done:
            return mask
        step_type, tier_idx = self._schedule[self._t]
        rng = _STEP_RANGE[step_type]
        mask[rng.start : rng.stop] = True
        if step_type == STEP_MODEL:
            mem = HARDWARE[self.tiers[tier_idx][0]]["memory_gb"]
            for j, m in enumerate(MODEL_NAMES):  # must fit at the most compact precision
                mask[MODEL_RANGE.start + j] = min_memory_gb(m, "int4") <= mem
        elif step_type == STEP_PREC:
            hw, model, _ = self.tiers[tier_idx]
            mem = HARDWARE[hw]["memory_gb"]
            for j, p in enumerate(PRECISION_NAMES):
                mask[PREC_RANGE.start + j] = min_memory_gb(model, p) <= mem
        return mask

    def _obs(self) -> np.ndarray:
        step_oh = np.zeros(5, dtype=np.float32)
        tier_oh = np.zeros(MAX_TIERS, dtype=np.float32)
        arch_oh = np.zeros(3, dtype=np.float32)
        stack_vec = np.zeros(MAX_TIERS * 12, dtype=np.float32)
        if not self._done:
            step_type, tier_idx = self._schedule[self._t]
            step_oh[step_type] = 1.0
            tier_oh[tier_idx] = 1.0
        if self.tier_count:
            arch_oh[self.tier_count - 1] = 1.0
        for i, (hw, model, prec) in enumerate(self.tiers):
            base = i * 12
            if hw is not None:
                stack_vec[base + HARDWARE_NAMES.index(hw)] = 1.0
            if model is not None:
                stack_vec[base + 4 + MODEL_NAMES.index(model)] = 1.0
            if prec is not None:
                stack_vec[base + 8 + PRECISION_NAMES.index(prec)] = 1.0
        return np.concatenate([self._profile_vec, step_oh, tier_oh, arch_oh, stack_vec])
