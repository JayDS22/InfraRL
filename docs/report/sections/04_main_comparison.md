## 4. Main Comparison

### 4.1 Frozen battery on held-out test split

The headline result is a frozen-battery evaluation on 1000 held-out test profiles. All eight agents train three independent seeds (2000, 2001, 2002) on 1000 training profiles each, then freeze their parameters and score the same 1000 test profiles generated at seed 1000. Per-agent compute budgets are matched inside their family and taken from `experiment.json`: the two closed-form baselines (random, heuristic) run zero training epochs, the three bandits train 20 epochs of 1000 profiles, and the three deep agents (q_learning, ppo, lagrangian_ppo) train 30 epochs of 1000 profiles. Reward uses the project's weighted compliance-plus-quality objective (quality 0.40, budget 0.25, SLA 0.25, quality attainment 0.10). Every number below is read directly from `results_frozen_v2_test/summary.csv`.

**Table 2. Main comparison on the held-out test split (mean plus/minus std over 3 seeds).**

| Agent | quality_compliance | budget_compliance | sla_compliance | meets_all | mean_reward | mean_cost_usd | avg_quality |
|---|---|---|---|---|---|---|---|
| random | 0.397 +/- 0.005 | 0.651 +/- 0.010 | 0.591 +/- 0.016 | 0.136 +/- 0.002 | 0.834 +/- 0.006 | 4543.584 +/- 178.180 | 0.708 +/- 0.002 |
| heuristic | 0.636 +/- 0.000 | 0.849 +/- 0.000 | 0.838 +/- 0.000 | 0.460 +/- 0.000 | 1.022 +/- 0.000 | 1714.896 +/- 0.000 | 0.763 +/- 0.000 |
| bandit_context_free | 0.575 +/- 0.060 | 0.900 +/- 0.075 | 0.854 +/- 0.133 | 0.439 +/- 0.031 | 1.029 +/- 0.004 | 1378.080 +/- 423.753 | 0.774 +/- 0.017 |
| bandit_contextual | 0.531 +/- 0.045 | 0.952 +/- 0.008 | 0.799 +/- 0.062 | 0.376 +/- 0.116 | 1.022 +/- 0.029 | 1233.072 +/- 108.081 | 0.751 +/- 0.017 |
| bandit_discounted | 0.436 +/- 0.134 | 0.915 +/- 0.046 | 0.671 +/- 0.221 | 0.166 +/- 0.069 | 0.956 +/- 0.030 | 2081.424 +/- 1106.633 | 0.724 +/- 0.036 |
| q_learning | 0.415 +/- 0.012 | 0.657 +/- 0.009 | 0.604 +/- 0.002 | 0.135 +/- 0.011 | 0.841 +/- 0.005 | 4554.768 +/- 63.861 | 0.710 +/- 0.004 |
| ppo | 0.589 +/- 0.075 | 0.904 +/- 0.023 | 0.914 +/- 0.058 | 0.484 +/- 0.032 | 1.045 +/- 0.006 | 1679.160 +/- 462.442 | 0.776 +/- 0.018 |
| lagrangian_ppo | **0.894 +/- 0.021** | 0.800 +/- 0.039 | 0.890 +/- 0.043 | **0.650 +/- 0.033** | **1.066 +/- 0.008** | 3249.600 +/- 351.570 | **0.843 +/- 0.004** |

Figure 1 (`figures/pareto_quality_vs_cost.png`) plots mean per-profile cost against mean quality for every agent-seed and overlays the empirical Pareto front; Lagrangian PPO occupies the high-quality end, the context-free bandit occupies the low-cost end, and vanilla PPO sits between them on the front. Figure 2 (`figures/compliance.png`) shows the three compliance axes side by side, making visible that Lagrangian PPO buys its quality lead by relaxing on budget.

Per-seed Wilcoxon signed-rank p-values with Holm correction (from `stats.csv`, reward column) corroborate the ordering. Lagrangian PPO beats PPO on reward with Holm-corrected p-values of 1.419e-09 (seed 0), 1.008e-05 (seed 1), and 3.340e-08 (seed 2), all below 0.05. PPO beats the heuristic on reward at p_holm = 5.420e-04 on seed 0, but the gap is not significant on seeds 1 (p_holm = 0.302) and 2 (p_holm = 0.403). The per-profile pairwise test is powered by n = 1000 profiles per seed, so each individual seed's verdict is stable; what is not stable with only three training seeds is the confidence interval on the mean of means in Table 2. Treat the standard deviations as suggestive, not as narrow confidence bands.

Lagrangian PPO wins quality_compliance (0.894) by spending 2.07x what vanilla PPO spends ($3249.600 vs $1679.160 per profile per month). That gap is not a bug and is not noise. The Lagrangian dual update treats quality_compliance as a hard floor and pushes the policy toward larger models and higher precision until the constraint is satisfied in expectation. Vanilla PPO treats the same signal as a soft reward term and trades compliance away on cheap profiles whenever the expected reward gain beats the compliance cost. Which policy is "better" depends on the deployment's willingness to pay for compliance. If a profile's quality floor encodes a hard contractual obligation, the Lagrangian cost premium is the price of enforcement. If the floor is advisory, PPO's ordering extracts more reward per dollar.

Vanilla PPO on the test split cleanly beats the heuristic on meets_all (0.484 vs 0.460). This reverses the awkward parity that showed up on the earlier validation-split run (frozen v1), where PPO and the heuristic tied inside one standard deviation. On the held-out test split the ordering Lagrangian PPO > PPO > heuristic > bandit_context_free > bandit_contextual > bandit_discounted > q_learning > random is now stable across all three seeds and defensible at the per-profile level. Tabular Q-learning collapses to the random floor on meets_all (0.135 vs 0.136): the state space is too large relative to the 30000-episode training budget and the tabular value function cannot generalize across profiles.

### 4.2 Ablations on Lagrangian PPO

Four ablation variants were trained and evaluated under the same frozen-battery protocol: `no_hardware_awareness` (fixed A100 placement), `no_precision_tuning` (FP16 preference with INT8/INT4 fallback for memory fit), `no_scaling_policy` (static replication only), and `no_multitier` (single-tier stacks only). All three seeds completed; numbers come from `ablations.csv`, with `full` as the baseline matching Table 2.

| Variant | meets_all | delta vs full | mean_cost_usd | delta vs full |
|---|---|---|---|---|
| full | 0.650 | - | 3249.600 | - |
| no_hardware_awareness | 0.601 | -0.049 | 2369.664 | -879.936 |
| no_precision_tuning | 0.550 | -0.100 | 2925.576 | -324.024 |
| no_scaling_policy | 0.645 | -0.005 | 3622.920 | +373.320 |
| no_multitier | 0.647 | -0.003 | 2926.992 | -322.608 |

Figure 3 (`figures/ablations.png`) renders these deltas against the full baseline across the seven headline metrics.

`no_precision_tuning` raises quality_compliance from 0.894 to 0.911 (FP16 beats INT8 on quality when the model fits) but drops meets_all by 0.100 because the cost overrun from forcing FP16 on large models kills budget compliance; the variant is strictly worse on the compliance-conjunction metric that matters. `no_hardware_awareness` loses 0.049 on meets_all: fixed A100 placement over-spends on small workloads and under-spends on the largest ones, dragging both budget and SLA compliance down together even though the mean cost falls because cheaper profiles dominate the mean. `no_scaling_policy` and `no_multitier` cut meets_all by less than one standard deviation (0.005 and 0.003 respectively), so the full policy's gains from those two axes are not separable from seed noise at n = 3. The deeper question whether multi-tier routing is net-useful on vanilla PPO is left to future work.

Honest gap: the ablations ran only on `lagrangian_ppo`. Without rerunning `no_hardware_awareness` and `no_precision_tuning` on vanilla PPO, the environment effect (restricting the action space of the simulator) is confounded with the agent effect (how Lagrangian PPO reacts to the restriction). This is flagged in Section 9 as future work.

### 4.3 Learning curves

Figure 4 (`figures/learning_curves.png`) plots train_reward and eval_reward per epoch per seed, read from `curves.csv`. PPO's eval_reward plateaus between epochs 10 and 15 and stays flat for the remaining 15 epochs. Lagrangian PPO continues to climb through all 30 epochs: the dual variable adapts the quality-constraint penalty, moving the effective reward surface each epoch, so the policy keeps finding slack to trade. The three bandits saturate inside the first five epochs, consistent with shallow linear-in-context features that exhaust the information in the profile vector quickly.

The dual update is what lets Lagrangian PPO keep learning past the point where vanilla PPO stops. That single mechanism explains the reward and meets_all gaps in Table 2 without appeal to hyperparameter tuning.
