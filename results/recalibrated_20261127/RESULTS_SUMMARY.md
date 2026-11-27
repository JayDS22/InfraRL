# InfraRL results after recalibration (2026-11-27)

Same protocol as `results_frozen_v2_test` (8 agents, 3 seeds, 1000 held-out test profiles,
seed 1000), run locally on CPU. The only change is the simulator constants, recalibrated
against public data (`docs/EXTERNAL_REFERENCE_CHECK.md`; diff in `recalibration.diff`).
Customer profiles, reward, agents and hyperparameters are unchanged.

## What the calibration changed
- Decode speed was 8-31x too fast; now 0.7-1.4x of measured llama.cpp speeds. Simulated
  latency rose by roughly an order of magnitude.
- A100 price $2.40 -> $2.69, CPU $0.40 -> $1.06; chatbot and code quality now follow Qwen2.5
  benchmarks; INT4 quality penalty 0.04 -> 0.021.
- **Feasibility collapsed:** only 34.5% of validation profiles can satisfy all three constraints
  with any single-tier stack (85.8% before). The profile generator's latency SLAs (e.g. chatbot
  0.4-2 s for full responses) are not achievable on real hardware for most stacks.
- Verification suite: 0 failures; 79/79 tests pass (one pinned reference value updated).

## Headline (test split, mean over 3 seeds)

| Method | Meets all | Quality ok | Budget ok | SLA ok | Reward | Before calibration (meets all) |
|---|---|---|---|---|---|---|
| **LinUCB (contextual bandit)** | **0.161** | 0.579 | 0.714 | **0.425** | **0.879** | 0.376 |
| D-LinUCB | 0.123 | 0.503 | 0.763 | 0.365 | 0.864 | 0.166 |
| PPO | 0.089 | 0.600 | **0.762** | 0.183 | 0.855 | 0.484 |
| Context-free bandit | 0.084 | 0.521 | 0.734 | 0.260 | 0.840 | 0.439 |
| Lagrangian PPO | 0.068 | **0.790** | 0.668 | 0.108 | 0.849 | **0.650** |
| Heuristic | 0.049 | 0.671 | 0.585 | 0.091 | 0.795 | 0.460 |
| Q-learning | 0.025 | 0.496 | 0.372 | 0.118 | 0.678 | 0.135 |
| Random | 0.022 | 0.472 | 0.369 | 0.119 | 0.673 | 0.136 |

Single-tier ceiling: about 0.35 of profiles are satisfiable at all. Validation split gives the
same ranking (LinUCB 0.184 meets-all, PPO 0.094, Lagrangian PPO 0.066).

Per-seed meets-all for LinUCB: 0.159 / 0.158 / 0.167 (best on every seed). On reward, LinUCB
beats every other method on every seed (Wilcoxon, Holm-corrected p < 0.001).
Mean cost is dominated by a few huge-replica outliers; median cost is $1,937/mo for every
learning agent (one A100 replica).

## Key findings
1. **The ranking flipped.** Under realistic latency, the contextual bandit (LinUCB) is the best
   method on both meets-all and reward. Lagrangian PPO drops from first to fifth: it still has
   by far the best quality compliance (0.79), but its quality-first constraint pushes it to large
   models whose latency misses the SLA (0.108).
2. **The earlier Lagrangian-PPO win was partly an artifact** of the placeholder simulator's
   unrealistically fast hardware, where large models were "free" on latency.
3. **The binding constraint is now latency**, not quality: SLA compliance is 0.09-0.43 for all methods.
4. **Online adaptation gain mostly vanished:** under full drift, D-LinUCB online vs static is
   +0.007 mean reward (was +0.039); under use-case drift alone, +0.001 to +0.002.
5. **Robustness:** LinUCB stays best under every +/-20% perturbation of quality, latency and
   cost (`sensitivity/sensitivity_meets_all.png`).

## Ablations (Lagrangian PPO)
| Variant | Quality ok | SLA ok | Budget ok | Cost/mo (mean) |
|---|---|---|---|---|
| Full | 0.790 | 0.108 | 0.668 | $9,014 |
| No hardware awareness | 0.792 | 0.252 | 0.588 | $10,086 |
| No multi-tier | 0.785 | 0.165 | 0.640 | $9,549 |
| No precision tuning | 0.815 | 0.068 | 0.605 | $15,729 |
| No scaling policy | 0.767 | 0.303 | 0.504 | $25,062 |
Removing precision tuning raises cost 75% and lowers SLA compliance; removing the scaling
policy (static, provision for peak) nearly triples cost.

## Caveats
- A10 and CPU decode speeds have no measured public reference (scaled from A100, unverified).
- RAG, agentic, batch and structured-extraction quality are still placeholders.
- Agents and hyperparameters were not retuned for the new simulator; PPO variants may improve
  with tuning. The profile SLA ranges were not changed.
