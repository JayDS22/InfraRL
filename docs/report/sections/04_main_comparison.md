## 4. Main Comparison

### 4.1 Frozen battery on held-out test split

Every number in this section is read directly from committed result
directories. Both runs share the same experimental protocol: eight agents
train three independent seeds (2000, 2001, 2002) on 1000 training
profiles each, then freeze their parameters and score the same 1000
held-out test profiles generated at seed 1000. Per-agent compute budgets
are matched inside their family: the two closed-form baselines (random,
heuristic) run zero training epochs, the three bandits train 20 epochs
of 1000 profiles, and the three deep agents (q_learning, ppo,
lagrangian_ppo) train 30 epochs of 1000 profiles. The reward uses the
weighted compliance objective defined in Section 3.1.

The two runs differ in exactly one dimension: the simulator's primitive
constants in `infrarl/simulator/hardware.py` and
`infrarl/simulator/models.py`. The first run uses the original
"structural" parameterization reviewed in the interim report; the second
uses the recalibrated constants refit against 29 public references
(Section 6.4). Everything else (customer profiles, action space, reward
function, seed regime, agent hyperparameters) is unchanged.

### 4.2 Pre-recalibration results (structural simulator)

**Table 2A. Pre-recalibration headline (mean plus/minus std over 3 seeds).**
Source: `results_frozen_v2_test/summary.csv` at commit 09a09f5.

| Agent | quality_compliance | budget_compliance | sla_compliance | meets_all | mean_cost_usd |
|---|---|---|---|---|---|
| random | 0.397 +/- 0.005 | 0.651 +/- 0.010 | 0.591 +/- 0.016 | 0.136 +/- 0.002 | 4544 +/- 178 |
| heuristic | 0.636 +/- 0.000 | 0.849 +/- 0.000 | 0.838 +/- 0.000 | 0.460 +/- 0.000 | 1715 +/- 0 |
| bandit_context_free | 0.575 +/- 0.060 | 0.900 +/- 0.075 | 0.854 +/- 0.133 | 0.439 +/- 0.031 | 1378 +/- 424 |
| bandit_contextual | 0.531 +/- 0.045 | 0.952 +/- 0.008 | 0.799 +/- 0.062 | 0.376 +/- 0.116 | 1233 +/- 108 |
| bandit_discounted | 0.436 +/- 0.134 | 0.915 +/- 0.046 | 0.671 +/- 0.221 | 0.166 +/- 0.069 | 2081 +/- 1107 |
| q_learning | 0.415 +/- 0.012 | 0.657 +/- 0.009 | 0.604 +/- 0.002 | 0.135 +/- 0.011 | 4555 +/- 64 |
| ppo | 0.589 +/- 0.075 | 0.904 +/- 0.023 | 0.914 +/- 0.058 | 0.484 +/- 0.032 | 1679 +/- 462 |
| **lagrangian_ppo** | **0.894 +/- 0.021** | 0.800 +/- 0.039 | 0.890 +/- 0.043 | **0.650 +/- 0.033** | 3250 +/- 352 |

Under the structural simulator Lagrangian PPO wins meets_all (0.650 vs
0.484 for the next-best agent, vanilla PPO) and quality_compliance
(0.894, the only agent above the 0.85 target), spending 2.07x what
vanilla PPO spends. Per-seed Wilcoxon signed-rank with Holm correction
on reward (`stats.csv`) corroborates this: Lagrangian PPO beats PPO on
all three seeds at p_holm < 1e-5. PPO beats the heuristic on seed 0 at
p_holm = 5.4e-4 but ties on seeds 1 and 2; the ordering is stable at
the per-profile level (n = 1000 paired) but std bands across seeds
are wide at n = 3.

### 4.3 Recalibrated results (production-anchored simulator)

**Table 2B. Post-recalibration headline (mean plus/minus std over 3 seeds).**
Source: `results/recalibrated_20261127/main_test/summary.csv`, run on
the simulator refit from 29 public references (Section 6.4). The protocol
and agent code are identical to Table 2A; only the simulator constants
changed.

| Agent | quality_compliance | budget_compliance | sla_compliance | meets_all | mean_cost_usd |
|---|---|---|---|---|---|
| random | 0.472 +/- 0.014 | 0.369 +/- 0.008 | 0.119 +/- 0.011 | 0.022 +/- 0.004 | 65054 +/- 7111 |
| heuristic | 0.671 +/- 0.000 | 0.585 +/- 0.000 | 0.091 +/- 0.000 | 0.049 +/- 0.000 | 10383 +/- 0 |
| bandit_context_free | 0.521 +/- 0.132 | 0.734 +/- 0.105 | 0.260 +/- 0.289 | 0.084 +/- 0.048 | 4209 +/- 1343 |
| **bandit_contextual** | 0.579 +/- 0.010 | 0.714 +/- 0.005 | 0.425 +/- 0.025 | **0.161 +/- 0.005** | 37317 +/- 21650 |
| bandit_discounted | 0.503 +/- 0.098 | 0.763 +/- 0.043 | 0.365 +/- 0.028 | 0.123 +/- 0.009 | 7179 +/- 4485 |
| q_learning | 0.496 +/- 0.009 | 0.372 +/- 0.011 | 0.118 +/- 0.006 | 0.025 +/- 0.005 | 67037 +/- 6330 |
| ppo | 0.600 +/- 0.083 | 0.762 +/- 0.042 | 0.183 +/- 0.108 | 0.089 +/- 0.042 | 4718 +/- 2188 |
| lagrangian_ppo | **0.790 +/- 0.021** | 0.668 +/- 0.025 | 0.108 +/- 0.014 | 0.068 +/- 0.006 | 9014 +/- 654 |

On the recalibrated simulator the ranking on meets_all flips. LinUCB
(bandit_contextual) wins at 0.161; PPO is second at 0.089; the
context-free bandit third at 0.084. Lagrangian PPO drops to fifth
(0.068) because its quality-first dual-update picks large, slow models
whose single-stream latency violates the SLA on most profiles
(sla_compliance 0.108). Lagrangian PPO still wins **raw quality
compliance** (0.790) and raw average quality (`avg_quality` 0.828); its
quality lead is preserved, but the newly-binding SLA constraint
collapses its operational performance.

The fully-satisfiable fraction over the 1000 test profiles drops from
85.8 percent on the structural simulator to 34.5 percent on the
recalibrated one. Many of the synthetic customer SLAs (for example a
chatbot requiring a 200-token completion in 400-1200 ms) are not
achievable by any 1- or 2-tier stack at measured single-stream decode
speeds (A100 at 30.5 tokens per second for 13B FP16, H100 at 34.3).
Section 9 flags "realistic customer SLA generator" as the natural
follow-up.

### 4.4 What the pivot means

The headline on Table 2A was not wrong given its inputs; it was wrong
as a production recommendation because the inputs themselves (vendor
batched-throughput numbers) did not reflect the single-stream workload
the simulator was supposed to model. Table 2B is the production answer.
Taken together, the two tables answer two different questions:

- **Table 2A answers:** which policy best trades quality for cost under
  a structural model where latency is cheap? Lagrangian PPO.
- **Table 2B answers:** which policy best trades quality, cost, and
  measured single-stream latency on real hardware? LinUCB.

Section 7's sensitivity analysis bridges the two. It established that
the policy ranking is stable under multiplicative perturbation of
simulator outputs within plus-or-minus 20 percent per axis. The
recalibration in Section 6.4 found the real latency miscalibration is
roughly 10x (not 20 percent). Section 7's analysis therefore correctly
predicted that a miscalibration of this magnitude would flip the
ranking; the empirical pivot in Table 2B is the confirmation. Section
8 develops this methodological finding in full.

### 4.5 Ablations on Lagrangian PPO (pre-recalibration)

Four ablation variants were trained and evaluated under the same
pre-recalibration protocol: `no_hardware_awareness` (fixed A100
placement), `no_precision_tuning` (FP16 preference with INT8/INT4
fallback for memory fit), `no_scaling_policy` (static replication
only), and `no_multitier` (single-tier stacks only). All three seeds
completed; numbers come from `results_frozen_v2_test/ablations.csv`,
with `full` as the baseline matching Table 2A.

| Variant | meets_all | delta vs full | mean_cost_usd | delta vs full |
|---|---|---|---|---|
| full | 0.650 | - | 3250 | - |
| no_hardware_awareness | 0.601 | -0.049 | 2370 | -880 |
| no_precision_tuning | 0.550 | -0.100 | 2926 | -324 |
| no_scaling_policy | 0.645 | -0.005 | 3623 | +373 |
| no_multitier | 0.647 | -0.003 | 2927 | -322 |

Figure 3 (`results_frozen_v2_test/figures/ablations.png`) renders these
deltas against the full baseline across the seven headline metrics.

`no_precision_tuning` raises quality_compliance from 0.894 to 0.911
(FP16 beats INT8 on quality when the model fits) but drops meets_all
by 0.100 because the cost overrun from forcing FP16 on large models
kills budget compliance. `no_hardware_awareness` loses 0.049 on
meets_all: fixed A100 placement over-spends on small workloads and
under-spends on the largest. `no_scaling_policy` and `no_multitier`
cut meets_all by less than one standard deviation at n = 3, so the
full policy's gains from those two axes are not separable from seed
noise. A parallel ablation of vanilla PPO (PR #4,
`results/final_20261125/ablate_ppo/`) is available in the repository
for cross-comparison and is noted in Section 9 as a candidate
confounding-check follow-up.

### 4.6 Learning curves

Figure 4 (`results_frozen_v2_test/figures/learning_curves.png`) plots
train and eval reward per epoch per seed, read from `curves.csv`.
Vanilla PPO's eval reward plateaus between epochs 10 and 15 and stays
flat through epoch 30. Lagrangian PPO continues to climb: the dual
variable adapts the quality-constraint penalty, moving the effective
reward surface each epoch, so the policy keeps finding slack to trade.
The three bandits saturate inside the first five epochs, consistent
with shallow linear-in-context features that exhaust the information
in the profile vector quickly. The dual update is what lets Lagrangian
PPO keep learning past the point where vanilla PPO stops; that
mechanism explains the quality_compliance lead in both Table 2A and
Table 2B.

### 4.7 Second-machine replication

`results/final_20261125/main_test/summary.csv` (PR #4, independent
CPU laptop run) replicates the Table 2A numbers within 0.015 meets_all
across all non-neural agents. Lagrangian PPO lands at 0.635 vs 0.650
and vanilla PPO at 0.468 vs 0.484 (torch nondeterminism across
machines), with the same ranking preserved. The pre-recalibration
Table 2A headline is reproducible on hardware outside the Zaratan
run.
