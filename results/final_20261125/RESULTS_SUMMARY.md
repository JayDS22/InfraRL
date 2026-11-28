# InfraRL final results (run 2026-11-25, local, Apple M4 Pro CPU)

Code: `main` @ 987fe72 (see `GIT_SHA.txt`). 3 seeds per method. Train on 1000 synthetic
profiles per seed (seeds 2000-2002); final scores on 1000 held-out **test** profiles (seed 1000).
`main_validation/` repeats everything on validation profiles (seed 3000) as a cross-check.

**Caveat:** customers are synthetic and quality/latency/cost come from the lookup-table simulator,
which is not calibrated against real hardware (calibration gate UNVERIFIED). These are real
training/evaluation runs in a simulated environment, not production measurements.

## Headline table (test set, mean over 3 seeds)

| Method | Meets all 3 | Quality ok | Budget ok | SLA ok | Mean cost/mo | Reward |
|---|---|---|---|---|---|---|
| **Lagrangian PPO** | **0.635** | **0.891** | 0.790 | 0.888 | $3,390 | **1.062** |
| PPO | 0.468 | 0.552 | 0.930 | **0.901** | $1,315 | 1.043 |
| Heuristic (rules) | 0.460 | 0.636 | 0.849 | 0.838 | $1,715 | 1.022 |
| Context-free bandit | 0.439 | 0.575 | 0.900 | 0.854 | $1,378 | 1.029 |
| LinUCB (contextual bandit) | 0.376 | 0.531 | **0.952** | 0.799 | **$1,233** | 1.022 |
| D-LinUCB (discounted) | 0.166 | 0.436 | 0.915 | 0.671 | $2,081 | 0.956 |
| Q-learning (tabular) | 0.135 | 0.415 | 0.657 | 0.604 | $4,555 | 0.841 |
| Random | 0.136 | 0.397 | 0.651 | 0.591 | $4,544 | 0.834 |

Targets from proposal: quality >= 0.85, budget >= 0.90, SLA >= 0.90, meets-all >= 0.75
(oracle ceiling ~0.84-0.86). Validation split gives the same ranking (Lagrangian PPO 0.659 meets-all).

Per-seed meets-all for Lagrangian PPO: 0.657 / 0.593 / 0.656 (best method on every seed).
Friedman test p ~ 0 on all seeds; Lagrangian PPO beats heuristic, PPO and all bandits on reward
(Wilcoxon, Holm-corrected p < 0.01) on 2 of 3 seeds, and on all 3 vs heuristic.

## Key findings
1. **Lagrangian PPO (constrained RL) is the best method**: +17.5 pts meets-all over the rule-based
   heuristic and +16.7 over plain PPO. It is the only method meeting the 0.85 quality-compliance target.
   Trade-off: it spends ~2x more ($3.4k vs $1.3-1.7k/mo) and budget compliance drops to 0.79.
2. **It learned per-use-case strategies**: 8-14 distinct stacks per seed, wins in every one of the 6 use cases
   (e.g. code generation 0.56 vs heuristic 0.33; agentic 0.40 vs 0.21).
3. **Plain PPO is cheap and SLA-safe but under-delivers on quality** (0.55); it uses only 3-9 stacks.
4. **Tabular Q-learning failed**: indistinguishable from random (state space too large for a table at 30 epochs).
5. **No method reaches the 0.75 meets-all target.**
6. Context-free bandit always picks one stack for everyone; it is a strong "one-size-fits-all" baseline.

## Per use case (meets-all, test)
| Use case | Lagrangian PPO | PPO | Heuristic | Context-free bandit |
|---|---|---|---|---|
| agentic | 0.40 | 0.22 | 0.21 | 0.22 |
| batch_inference | 0.79 | 0.65 | 0.63 | 0.59 |
| chatbot | 0.61 | 0.52 | 0.42 | 0.46 |
| code_generation | 0.56 | 0.29 | 0.33 | 0.29 |
| rag | 0.65 | 0.45 | 0.45 | 0.43 |
| structured_extraction | 0.79 | 0.67 | 0.71 | 0.63 |

## Ablations (Lagrangian PPO, test, mean over 3 seeds)
| Variant | Meets all | SLA ok | Cost/mo |
|---|---|---|---|
| Full | 0.635 | 0.888 | $3,390 |
| No precision tuning (FP16 only) | 0.560 | **0.722** | $3,006 |
| No hardware awareness (A100 only) | 0.597 | 0.836 | $2,386 |
| No multi-tier | 0.645 | 0.851 | $2,882 |
| No scaling policy (static) | 0.642 | 0.900 | $3,451 |

Precision tuning matters most (removing it drops SLA compliance 0.89 -> 0.72). Hardware awareness
is second. Removing multi-tier or scaling made no significant difference (within seed noise, std ~0.05).
PPO ablations (`ablate_ppo/`) show restricting the action space actually *helps* plain PPO
(0.47 -> 0.53), i.e. PPO is under-exploring the full space.

## Online adaptation under workload drift (4 weeks, 250 profiles/week, 3 seeds)
| Bandit | Drift type | Static reward | Online reward | Gain |
|---|---|---|---|---|
| LinUCB | use-case mix only | 0.993 | 0.991 | -0.002 |
| D-LinUCB | use-case mix only | 0.914 | 0.918 | +0.004 |
| LinUCB | full (mix + accuracy + budget + priorities) | 0.913 | 0.920 | +0.007 |
| **D-LinUCB** | **full** | **0.837** | **0.875** | **+0.039** |

Under heavy drift, the static recommendation decays (0.91 -> 0.79 by week 2-3) while the online
discounted bandit holds ~0.85-0.86. Under mild drift, adapting online gives no benefit.

## Which figure for which slide
- Evaluation / Results: `main_test/figures/compliance.png`, `pareto_quality_vs_cost.png`,
  `heatmap_meets_all_by_use_case.png`, `learning_curves.png`
- Ablations: `ablate_lagrangian_ppo/figures/ablations.png`
- Online adaptation: `drift_discounted_full/drift.png`

## Not done / limitations
- DQN never implemented (`agents/dqn_agent.py` empty); online `contextual_bandit.py` empty (drift uses agents/bandit.py).
- Simulator uncalibrated; profiles synthetic.
- Only 3 seeds; Lagrangian PPO budget compliance (0.79) below the 0.90 target.
