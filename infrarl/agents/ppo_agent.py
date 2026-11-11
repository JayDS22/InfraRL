"""Hand-rolled PPO for the sequential stack-construction MDP.

Clipped-surrogate PPO (Schulman et al., 2017) with GAE (Schulman et al., 2016).
Per-step invalid-action masking follows Huang & Ontanon (2022): the Categorical
distribution is built over legal actions only, by filling masked logits with a
large negative constant before softmax. The log-prob and entropy of the chosen
action therefore ignore illegal arms, and gradients never flow to them.

Flow:
  select_action buffers (obs, mask, action, logp, value) into self._pending.
  update consumes self._pending plus the Transition's reward/terminated flag
  and appends to the rollout buffer. When the buffer reaches rollout_size at
  an episode boundary, we run update_epochs passes of minibatch PPO updates
  and clear the buffer.

Registered through infrarl.agents.registry._try_register("ppo", ...) which
succeeds the moment this module imports cleanly (torch must be installed).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from ..simulator.environment import N_ACTIONS, OBS_DIM
from .base import BaseAgent, Transition

_NEG_INF_LOGIT = -1e9


class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.pi_head = nn.Linear(hidden, n_actions)
        self.v_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        return self.pi_head(h), self.v_head(h).squeeze(-1)


def _masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    return Categorical(logits=logits.masked_fill(~mask, _NEG_INF_LOGIT))


class PPOAgent(BaseAgent):
    name = "ppo"

    def __init__(
        self,
        seed: int = 0,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip: float = 0.2,
        rollout_size: int = 2048,
        update_epochs: int = 4,
        minibatch_size: int = 256,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ):
        # ponytail: global torch seed, one agent per process is fine. Per-instance
        # torch.Generator if we ever run multiple PPO agents in the same interpreter.
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.net = PolicyValueNet()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.rollout_size = rollout_size
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self._buf: list[tuple] = []
        self._pending: tuple | None = None

    def select_action(self, obs: np.ndarray, mask: np.ndarray, explore: bool = True) -> int:
        o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        m = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.net(o)
            dist = _masked_categorical(logits, m)
            if explore:
                action = dist.sample()
            else:
                action = logits.masked_fill(~m, _NEG_INF_LOGIT).argmax(dim=-1)
            logp = dist.log_prob(action)
        a = int(action.item())
        self._pending = (obs.copy(), mask.copy(), a, float(logp.item()), float(value.item()))
        return a

    def update(self, t: Transition) -> None:
        if self._pending is None:
            return
        obs, mask, action, logp, val = self._pending
        assert action == t.action, "PPO transition action does not match the sampled action"
        self._buf.append((obs, mask, action, logp, val, float(t.reward), bool(t.terminated)))
        self._pending = None
        if len(self._buf) >= self.rollout_size and t.terminated:
            self._learn()
            self._buf = []

    def _learn(self) -> None:
        obs = np.stack([x[0] for x in self._buf]).astype(np.float32)
        masks = np.stack([x[1] for x in self._buf]).astype(bool)
        acts = np.array([x[2] for x in self._buf], dtype=np.int64)
        logp_old = np.array([x[3] for x in self._buf], dtype=np.float32)
        vals = np.array([x[4] for x in self._buf], dtype=np.float32)
        rews = np.array([x[5] for x in self._buf], dtype=np.float32)
        dones = np.array([x[6] for x in self._buf], dtype=bool)

        advantages = np.zeros_like(rews)
        gae = 0.0
        next_v = 0.0  # terminal bootstrap: last transition is terminal by construction
        for i in reversed(range(len(rews))):
            non_terminal = 0.0 if dones[i] else 1.0
            delta = rews[i] + self.gamma * next_v * non_terminal - vals[i]
            gae = delta + self.gamma * self.lam * non_terminal * gae
            advantages[i] = gae
            next_v = vals[i]
        returns = advantages + vals
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t = torch.as_tensor(obs)
        masks_t = torch.as_tensor(masks)
        acts_t = torch.as_tensor(acts)
        logp_old_t = torch.as_tensor(logp_old)
        adv_t = torch.as_tensor(advantages)
        ret_t = torch.as_tensor(returns)

        n = len(self._buf)
        idx = np.arange(n)
        for _ in range(self.update_epochs):
            self.rng.shuffle(idx)
            for start in range(0, n, self.minibatch_size):
                b = idx[start : start + self.minibatch_size]
                bi = torch.as_tensor(b, dtype=torch.long)
                logits, values = self.net(obs_t[bi])
                dist = _masked_categorical(logits, masks_t[bi])
                lp_new = dist.log_prob(acts_t[bi])
                entropy = dist.entropy().mean()
                ratio = torch.exp(lp_new - logp_old_t[bi])
                surr1 = ratio * adv_t[bi]
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_t[bi]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (values - ret_t[bi]).pow(2).mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.opt.step()
