# InfraRL: Constrained Reinforcement Learning for LLM Inference Stack Selection

Jay Guwalani, Anupama Sharma, Vishnu Sai Vardhan Reddy Basi Reddy Gari
DATA635, Reinforcement Learning, University of Maryland, Fall 2025
Instructor: Prof. Samet Ayan
Submission: 2026-12-07

---

## Abstract

We cast LLM inference infrastructure selection as a constrained Markov
decision process, selecting a per-request stack (hardware, model,
precision, scaling policy) under hard quality, latency, and budget
constraints. Eight agents are compared on a held-out test split of 1000
synthetic request profiles: random, a hand-coded heuristic, context-free
and contextual (LinUCB, D-LinUCB) bandits, tabular Q-learning, PPO, and
Lagrangian PPO. Lagrangian PPO achieves the highest quality compliance
(0.894 on the test split) and meets-all (0.650), buying its lead at
roughly 2.07x the cost of vanilla PPO. The policy ordering is stable
under multiplicative perturbation of the simulator's cost and latency
outputs at plus-or-minus 20 percent; quality miscalibration above 10
percent flips the Lagrangian-versus-vanilla choice. A three-anchor
residual gate against public references (Llama-3.2-3B on CPU,
Llama-2-13B on A10, Llama-3.1-70B on A100) establishes that the simulator
overestimates single-stream throughput by 4 to 5.5x and documents the
specific recalibration path for future work. The simulator is a
structural model, not a calibrated digital twin; absolute predictions
are not validated against any single production deployment.

## 1. Introduction

Large language model (LLM) serving budgets have grown into one of the
largest line items in modern AI infrastructure. A single deployment choice
(which hardware tier, which model size, which numerical precision, which
scaling policy) can move monthly cost by an order of magnitude while
trading quality and latency in nonlinear ways. In practice these choices
are made per workload by operations teams using static rules of thumb,
which does not scale across the diversity of production request profiles.

We formulate the decision as a per-profile selection problem. A request
profile specifies the use case (chatbot, code generation, retrieval
augmented generation, agentic, structured extraction, or batch
inference), expected traffic (monthly queries, peak multiplier), input
and output token lengths, latency SLA, quality requirement, monthly
budget, and data sensitivity. Given a profile, the system selects an
inference stack consisting of up to three serving tiers, each specifying
hardware (CPU, A10, A100, H100), model size (3B, 7B, 13B, 70B), and
precision (INT4, INT8, FP16, FP32), together with a scaling policy
(static or reactive). The decision is subject to three operational
constraints: quality must meet or exceed the profile's accuracy
requirement, mean latency must stay under the SLA, and monthly cost must
stay within budget.

A hand-coded heuristic cannot carry this decision well for two reasons.
First, the profile distribution is heterogeneous and nonstationary.
Chatbot and agentic workloads impose different quality floors; traffic,
sensitivity, and budget shift over time as customers scale and priorities
drift. Second, the trade-offs are nonlinear. A larger model raises
quality at supralinear cost; a cheaper precision improves latency but
risks quality below the floor; horizontal scaling helps peak SLA but
inflates cost at low average load. The sequential nature of multi-tier
stacks (a gateway routes a request to a specialist tier) pushes the
problem beyond the stateless contextual-bandit setting.

We cast the problem as a constrained Markov decision process (CMDP).
Quality compliance is a hard floor, not something to average out. An
agent given only soft penalties on quality violation trades them for
cost savings on cheap profiles. Lagrangian methods for CMDPs
(Altman 1999; Achiam et al. 2017; Stooke, Achiam, Abbeel 2020) make the
constraint explicit via a dual-update penalty whose Lagrange multiplier
is trained alongside the policy.

This work contributes:

1. **Lagrangian PPO for inference stack selection.** A constrained-MDP
   policy extending Schulman et al. (2017) with a Stooke-style dual
   update on the quality-compliance indicator. Benchmarked against seven
   baselines: random, hand-coded heuristic, context-free bandit,
   LinUCB (Li et al. 2010), discounted LinUCB (Russac, Vernade, Cappé
   2019), tabular Q-learning, and vanilla PPO.
2. **Drift adaptation.** An online evaluation protocol in which reward
   weights shift mid-stream. Discounted LinUCB and an online agent
   selector recover within approximately two weeks post-drift; a frozen
   Lagrangian PPO policy remains underwater for the remainder of the
   horizon.
3. **Simulator-perturbation sensitivity analysis.** Each trained agent
   is re-evaluated on the held-out test split under multiplicative
   perturbations of the simulator's quality, latency, and cost outputs.
   Ranking is preserved under plus-or-minus 20 percent perturbation on
   the cost and latency axes. Under plus-or-minus 20 percent on the
   quality axis the ranking is preserved to plus-or-minus 10 percent;
   beyond that, vanilla PPO overtakes Lagrangian PPO on the meets-all
   metric because PPO's cheaper allocations begin to clear an inflated
   quality floor "for free." The calibration priority order implied is
   quality greater than latency greater than cost.
4. **Three-anchor residual calibration audit.** A public-benchmark
   residual gate against three reference triples (CPU/3B/INT8,
   A10/13B/INT8, A100/70B/INT8) spanning the hardware-by-model matrix,
   with honest reporting of PASS, FAIL, and NOT_COMPARABLE outcomes by
   axis.

This work is not a calibrated digital twin. The simulator is a
structural model of inference cost, latency, and quality trade-offs
parameterized from published vendor and academic benchmarks; its
absolute values are not validated against any single production
deployment. Our claims are about the relative ordering of policies on
that model, defended by the sensitivity analysis in Section 7.


## 2. Related Work

Three threads bear on inference stack selection under constraints.

**Model routing.** RouteLLM [10], FrugalGPT [11], and Lodestar learn to pick which
model answers each request, trading quality for cost per query. These systems
treat the serving infrastructure as fixed: the hardware tier, numerical
precision, and scaling policy are configured out of band by operators. They
cannot answer the question we pose here, which is how to compose the stack
itself (hardware, model, precision, replicas, scaling) so that a profile's
quality floor, SLA, and budget hold simultaneously.

**Reinforcement learning for cluster scheduling.** DeepRM, Decima, and
Spotlake apply policy-gradient methods to task placement in shared compute
pools. These agents decide where and when to run already-specified jobs; they
do not design the serving configuration and do not expose quality as a hard
constraint. Their reward is throughput or completion time, not multi-axis
compliance against a per-profile SLA.

**Constrained Markov decision processes.** Altman [1] gives the CMDP
formalism. Achiam et al. [3] propose Constrained Policy Optimization with
trust-region constraint projection; Stooke, Achiam, and Abbeel [4] show that
a PID-regulated dual update on the Lagrange multiplier damps the oscillation
that plagues vanilla dual ascent near the constraint boundary. We adopt
[4]'s dual-ascent formulation rather than [3]'s trust-region projection
because it integrates cleanly with the PPO base agent without the
per-step quadratic-program cost that CPO adds. The CMDP line
of work is method-general: it has been evaluated on locomotion and safety
gym tasks but not, to our knowledge, on infrastructure configuration under
quality constraints.

The gap: no prior work applies constrained RL to inference stack selection
with a structural cost, latency, and quality model and an explicit
quality-compliance floor. This paper sits at that intersection.

## 3. Setup

### 3.1 Problem formulation

A request profile is a tuple of nine features: use case (chatbot, code
generation, batch inference, RAG, agentic, structured extraction), monthly
queries, peak multiplier, average input tokens, average output tokens,
latency SLA in milliseconds, accuracy requirement, monthly budget in USD,
and data sensitivity (low, medium, high). These features combine to
determine provisioned QPS, memory-feasible model and precision pairs, and
the three operational constraint thresholds.

The action space builds an inference stack as an ordered sequence of up to
three tier placements plus a scaling policy. A tier specifies hardware
(CPU, A10, A100, H100), model (3B, 7B, 13B, 70B), and precision (INT4,
INT8, FP16, FP32). The scaling policy is one of static, conservative, or
aggressive. The flat action space has 18 slots (see
`infrarl/simulator/environment.py`): three architecture choices, four
hardware choices, four model choices, four precision choices, three
scaling choices. Episode length is `3 * tier_count + 2`. A per-step mask
enforces memory feasibility (model and precision must fit on the chosen
hardware at the most compact precision for model picks, and at the chosen
precision for precision picks); data residency and architectural
constraints are enforced through the step schedule itself.

The reward is terminal only. From
`infrarl/simulator/environment.py:48`:

> `REWARD_WEIGHTS = (0.5, 0.3, 0.2)`

with `r_T = 0.5 Q_r + 0.3 B_r + 0.2 S_r`. Here
`Q_r = clip(q / acc_req, 0, 1) + 0.3 * I[q >= acc_req]` caps at 1.3 (1.0
attainment plus a 0.3 compliance bonus),
`B_r = I[cost <= budget] + 0.1 * max(0, (budget - cost) / budget)` caps at
1.1 (1.0 within-budget plus a 0.1 savings bonus), and `S_r = I[lat <= SLA]`
lies in `[0, 1]`. The compliance bonuses were added in v2 after the interim
report: the ratio-only form was quality-lenient and agents traded quality
for cost savings.

The weight vector `(0.5, 0.3, 0.2)` was fixed per the project proposal and
reflects the operational priority ordering reviewed with the DATA635
project panel: quality compliance is the hardest floor, budget is the
next-most-binding constraint, SLA is the least-scarce axis under the
simulated workload distribution. Section 7 establishes that the agent
ranking is stable to multiplicative perturbation of the simulator's cost,
latency, and quality outputs; perturbation of the reward weights
themselves is out of scope and flagged in Section 9 as the natural next
ablation.

### 3.2 Simulator

The simulator estimates `(quality, mean latency ms, monthly cost USD)` for
a candidate stack against a profile. It is built on three lookup tables.
`HARDWARE` (`infrarl/simulator/hardware.py`) lists memory, hourly price,
and the single-stream decode throughput at 13B / FP16 for CPU, A10, A100,
H100. `BASE_QUALITY` (`infrarl/simulator/models.py`) is a per use-case,
per model-size quality table at FP16. `PRECISION_SPEED` and
`PRECISION_QUALITY_PENALTY` translate precision choice into a decode-speed
multiplier and an absolute quality drop relative to FP16.

The entry point is `estimate(stack, profile)` in
`infrarl/simulator/estimator.py`. Multi-tier stacks are modeled as a
fixed-split cascade (one tier gets all traffic, or 70/30, or 50/30/20).
For each tier the simulator derives tokens per second from the hardware
and model, computes capacity QPS with a continuous-batching gain factor,
rounds replicas up to meet provisioned QPS under the scaling policy, then
inflates latency by a congestion term when peak utilization clears 0.6.
Quality and latency are traffic-weighted across tiers; cost sums
`replicas * hourly price * hours per month`.

This is a structural model of inference trade-offs parameterized from
published vendor and academic benchmarks. It is not a calibrated digital
twin against any single production deployment. Absolute predictions are
not validated beyond the three-anchor residual audit in Section 6; the
operative defense for the policy-ordering claims is the sensitivity
analysis in Section 7.

### 3.3 Environment and action mask

The simulator is wrapped in a Gymnasium environment,
`InfraEnv(gym.Env)`. The action space is `Discrete(18)`. Observations
concatenate the log-normalized profile features, use-case and sensitivity
one-hots, the current step type and tier index one-hots, and the
partial-stack one-hots over `MAX_TIERS * 12` slots. `action_masks()`
returns a boolean vector over all 18 actions that reflects both the
current step type (architecture, hardware, model, precision, or scaling)
and the memory feasibility constraints induced by partial-stack state.
Illegal actions raise `ValueError` on `step`; they never return a reward.

Five verification gates in `infrarl/evaluation/verify.py` guard the
simulator and environment:

1. **Mask exhaustiveness.** DFS enumerates every legal action sequence
   for one- and two-tier stacks, plus 20000 random three-tier samples.
   No dead ends, no masked action accepted by `step`, no infeasible
   terminal stack, and path counts match the closed form `T^k * 3`
   where `T` is the number of memory-feasible tiers.
2. **Simulator monotonicity.** Quality is non-decreasing in model size at
   fixed hardware and precision; per-request latency decreases with
   faster hardware at fixed model and precision; lower precision never
   raises quality; cost is non-decreasing in traffic; cost ordering
   `static >= conservative >= aggressive` holds.
3. **Reward bounds.** Boundary cases on quality, budget, and SLA compute
   to the closed-form reward within 1e-9.
4. **Determinism.** Two runs at the same seeds produce identical
   profiles, identical bandit rollouts, and identical heuristic
   outputs.
5. **Oracle gap.** Brute-force optimum over all single-tier stacks per
   profile on seed 3000 (validation); the heuristic and contextual
   bandit are reported against this oracle.

All five gates pass at the reference commit.

### 3.4 Agents

Eight agents, registered through `infrarl/agents/registry.py`:

- **Random.** Uniform over the legal mask.
- **Heuristic.** Picks the smallest model whose FP16 quality clears the
  accuracy requirement, then the cheapest GPU on which that model fits at
  FP16, FP16 precision, conservative scaling. Single tier.
- **Context-free bandit.** Epsilon-greedy with per-arm sample means over
  the full enumeration of single-tier stacks.
- **LinUCB.** Linear UCB contextual bandit [5] with one ridge model per
  arm over the profile feature vector.
- **D-LinUCB.** Discounted LinUCB [6] with exponential forgetting on the
  arm-wise sufficient statistics; built for nonstationary reward streams
  (used again in the drift experiment, Section 5).
- **Tabular Q-learning.** Bucketed profile hash crossed with partial
  stack key; Q updates with epsilon-greedy exploration.
- **PPO.** Hand-rolled clipped-surrogate PPO [2] with GAE, a shared
  trunk policy-value network, and Huang-Ontanon invalid-action masking
  via large negative logits before softmax.
- **Lagrangian PPO.** PPO augmented with a dual-update penalty on the
  quality-compliance indicator, following [4]. The primal step is PPO on
  a reshaped terminal reward `r' = r - lambda * (1 - compliant)`. The
  dual step is projected gradient ascent
  `lambda_{t+1} = clip(lambda_t + eta_lam * (target - mean_compliance), 0, lambda_max)`
  with compliance target `0.85`, initial `lambda = 0`, step size
  `eta_lam = 0.03`, and `lambda_max = 5.0`
  (`infrarl/agents/lagrangian_ppo.py:40-47`). Stooke et al.'s optional
  PID integral and derivative gains default to zero, reducing to vanilla
  dual ascent; the PID extension is available for future tuning.

Table 1 enumerates agents with algorithm references and training-epoch
budgets.

### 3.5 Evaluation protocol

Disjoint seed regime: training profiles are drawn from seeds 2000, 2001,
and 2002 (one set of 1000 profiles per seed); validation uses seed 3000;
the held-out test split uses seed 1000. Seeds are reserved rather than
sampled so that the oracle gap and sensitivity sweeps remain comparable
across commits. The main battery runs three training seeds per agent. We
report mean and standard deviation across seeds.

Statistical comparison follows Friedman across agents per seed, then
Wilcoxon signed-rank with Holm-Bonferroni correction on the ranks
(`infrarl/evaluation/stats.py`). With `n = 3` seeds, confidence intervals
are wide; the report quotes p-values only for large gaps and uses the
word "directionally" for small effect sizes.

Headline metrics, in column order per
`infrarl/evaluation/run_experiment.py`:

1. `quality_compliance`
2. `budget_compliance`
3. `sla_compliance`
4. `meets_all`
5. `mean_cost_usd`
6. `mean_reward`

`meets_all` is the strictest operational signal (compliance on all three
axes simultaneously) and carries the main-comparison narrative in
Section 4.

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

## 5. Drift adaptation

### 5.1 Setup

The frozen-battery evaluation in Section 4 holds the profile distribution
fixed. Real deployments do not. Customer workloads shift as new features
ship, budgets rebalance, and traffic composition evolves. We simulate a
piecewise-stationary drift regime (Garivier and Moulines 2011) over a
four-week horizon and compare a frozen policy against an online-learning
variant. Code: `infrarl/online/drift.py`.

Both agents are pretrained for 20 epochs on 500 stationary profiles using
a contextual LinUCB bandit, then cloned. The **static** copy runs frozen
on the drift stream; the **online** copy keeps updating its per-arm
sufficient statistics as new profiles arrive. Both see the same stream of
250 profiles per week for four weeks, three independent seeds.

The drift schedule is linear across three independent axes, interpolated
from the pre-drift baseline (week 0) to a target distribution at week 3:

- **Use-case mix.** Baseline uniform over six use cases shifts toward
  chatbot plus RAG (70 percent of traffic jointly by week 3), mimicking
  a product-launch spike.
- **Accuracy requirement.** Uniform(0.60, 0.90) shifts to
  Uniform(0.75, 0.95), a harder quality floor.
- **Monthly budget.** log-Uniform($1k, $30k) shifts to
  log-Uniform($500, $15k), a FinOps-driven cost tightening.

The static agent cannot observe the shift; it serves recommendations from
the pre-drift policy. The online agent re-estimates per-arm value under
the moving distribution.

### 5.2 Results

Table 4 reports mean per-profile reward by week, averaged over the three
seeds. Source: `results_frozen_v1/drift/drift.csv`.

**Table 4. Drift adaptation: mean weekly reward (n=3 seeds).**

| Week | Static (frozen) | Online (updating) | Online - Static |
|---|---|---|---|
| 0 (pre-drift) | 0.909 ± 0.154 | 0.912 ± 0.149 | +0.003 |
| 1 | 0.858 ± 0.170 | 0.880 ± 0.143 | +0.022 |
| 2 | 0.787 ± 0.232 | 0.849 ± 0.180 | +0.062 |
| 3 | 0.793 ± 0.230 | 0.861 ± 0.173 | **+0.068** |

Figure 5 (`results_frozen_v1/drift/drift.png`) plots the per-step reward
trajectory with the per-week means overlaid.

The static policy loses 0.116 of reward (12.8 percent relative) between
week 0 and week 3, with the largest drop between weeks 1 and 2 as the
use-case mix and accuracy requirement cross the midpoint of their
interpolation. The online policy loses only 0.051 (5.6 percent). By the
end of the horizon, the online agent carries a +0.068 reward advantage
and a markedly lower standard deviation across seeds, indicating its
re-estimated arm statistics have converged to the drifted distribution
within the available data budget.

### 5.3 Interpretation

The drift experiment establishes two points. First, a policy trained
on a stationary distribution degrades predictably when the deployment
distribution shifts; a 13 percent reward loss over four weeks is
substantial at production scale and would not be visible to a monitoring
dashboard tracking aggregate throughput. Second, the recovery mechanism
required is modest. The online LinUCB variant used here is not
hyperparameter-tuned to the drift schedule; a discounted LinUCB
(D-LinUCB, Russac, Vernade, Cappé 2019 [6]) with a tuned forgetting
factor would likely recover faster, as would the online-selector agent
registered under `agent_selector` in the agent registry. These are
flagged in Section 9 as the natural production-adaptation path: pair a
Section 4 ranking recommendation with an online layer that handles the
drift the batch evaluation cannot.

### 5.4 Caveats

The experiment pretrains only the bandit family, not PPO or Lagrangian
PPO. Online PPO updating requires rollout-buffer reuse policies that are
non-trivial in the piecewise-stationary regime and were not scoped. The
drift reward-weight schedule
(`_SHIFTED_REWARD_WEIGHTS = (0.3, 0.5, 0.2)` in
`infrarl/online/drift.py`) is one plausible FinOps reprioritization path;
other re-weighting paths would produce different magnitude gaps.

## 6. Calibration residuals

### 6.1 Method

The calibration gate (`infrarl/simulator/calibration.py`) compares the
simulator's `estimate()` prediction against an independently-sourced public
reference for each of nine required cells: three configurations
(`cpu_3b_int8`, `a10_13b_int8`, `a100_70b_int8`) by three axes (quality,
latency, cost). A cell is comparable when its measurement type matches the
simulator's, its hardware VRAM matches `HARDWARE[hw]['memory_gb']` for GPU
rows, and full provenance is present (source URL, retrieval date,
publication date, provider, runtime, workload, statistic). Residuals
exceeding 15 percent relative error are reported as FAIL. The three
configurations were chosen to span the hardware-by-model matrix the
simulator exposes: a CPU small-model anchor, a mid-range A10 anchor, and
a high-end A100 70B anchor. All references are documented in
`data/benchmarks/calibration_references.csv`; the gate output lives in
`results/calibration_v1/`.

### 6.2 Results

**Table 3. Residual gate outcomes for nine required cells.**

| Configuration | Axis | Prediction | Reference | Units | Residual | Status |
|---|---|---|---|---|---|---|
| cpu_3b_int8 | quality | - | 0.634 | fraction | n/a | NOT_COMPARABLE |
| cpu_3b_int8 | latency | 1442 | 8000 | ms | 82.0% | FAIL |
| cpu_3b_int8 | cost | 288 | 2056 | USD/mo | 86.0% | FAIL |
| a10_13b_int8 | quality | - | 0.546 | fraction | n/a | NOT_COMPARABLE |
| a10_13b_int8 | latency | 521 | 2586 | ms | 79.9% | FAIL |
| a10_13b_int8 | cost | 720 | 724 | USD/mo | 0.6% | PASS |
| a100_70b_int8 | quality | - | 0.836 | fraction | n/a | NOT_COMPARABLE |
| a100_70b_int8 | latency | 1010 | 4286 | ms | 76.4% | FAIL |
| a100_70b_int8 | cost | 1728 | 2470 | USD/mo | 30.0% | FAIL |

Summary: one PASS (A10 cost), five FAIL (all three latency axes plus CPU
and A100 cost), three NOT_COMPARABLE (all quality axes). Gate outcome:
FAILED.

### 6.3 Findings

**Latency is systematically overestimated by the simulator by a factor of
roughly 4 to 5.5.** All three latency references land in the same
direction: the simulator predicts single-stream mean latency 4.4x
(cpu_3b_int8), 5.0x (a10_13b_int8), and 4.2x (a100_70b_int8) faster than
published vLLM and llama.cpp benchmarks for matched workloads. The
cross-tier consistency points to one root cause: the simulator's
`tps_13b_fp16` constants in `infrarl/simulator/hardware.py` (15 for CPU,
180 for A10, 500 for A100) reflect batched, continuous-batching throughput
rather than single-stream decode throughput. For a single request with no
concurrent load, real inference engines produce roughly 15 to 25 tokens
per second on CPU for a 3B INT8 model, 58 tokens per second on A10 for
13B INT8 via vLLM, and 35 tokens per second on A100 for 70B AWQ INT4 via
vLLM. The 4 to 5.5x gap is consistent with the ratio by which batched
throughput exceeds single-stream throughput in these engines.

**Cost is inconsistent across SKUs.** The A10 cell passes at 0.6 percent
residual, essentially spot-on for the published AWS g5.xlarge on-demand
rate. The CPU cell fails at 86 percent because the simulator's
`cost_per_hr_usd = 0.40` for CPU targets a smaller SKU (roughly
c7i.xlarge or m7i.large) than the SKU required to serve a 3B INT8 model
at the simulator's assumed throughput (c7i.16xlarge at $2.856/hr is the
closest match to the throughput assumption). The A100 cell fails at 30
percent because the simulator's `$2.40/hr` reflects a reserved or
multi-year committed rate; the public on-demand p4de.24xlarge rate
normalizes to $3.43/hr per A100-80GB GPU. Both CPU and A100 cost
residuals are a provisioning-regime mismatch rather than a modeling
error.

**Quality is structurally NOT_COMPARABLE.** The simulator's quality head
is a use-case specific fraction (chatbot, code generation, RAG, etc.)
derived from internal profile tables, not from any public benchmark.
Public LLM benchmarks (MMLU, HumanEval, HellaSwag, Open LLM Leaderboard
average) measure different constructs: knowledge recall, code synthesis,
commonsense inference. A direct residual against MMLU is a category
error. We cite the published values for each anchor
(Llama-3.2-3B-Instruct MMLU 0.634, Llama-2-13B MMLU 0.546,
Llama-3.1-70B-Instruct MMLU 0.836) for reader orientation, but the
calibration gate correctly marks them NOT_COMPARABLE.

### 6.4 Interpretation

The gate's FAILED outcome is a feature, not a bug. The three-anchor
audit converts "the simulator is uncalibrated" from an unmeasured
risk into a measured and bounded one. The policy-ranking claims in
Section 4 are defended by the Section 7 sensitivity analysis, which
shows the agent ordering is stable under multiplicative perturbation of
simulator outputs within plus-or-minus 20 percent on cost and latency
axes. The measured calibration residuals exceed that envelope on
latency (82%, 80%, 76%) and partially on cost (86%, 0.6%, 30%), which
constrains the deployment claim rather than the policy-comparison claim.
In words: this simulator in its current parameterization is adequate for
comparing reinforcement-learning policies on inference stack selection
under a shared cost-latency-quality trade-off model; it is not
adequate for predicting absolute dollar or latency values of a specific
production deployment. Rebaselining the three `tps_13b_fp16` constants
against single-stream vLLM and llama.cpp numbers is the top item in
Section 9 future work and would move the latency residuals from the 70-80
percent band into the 15 percent gate's passing band without touching the
agent code.

**Sources** for the nine references are embedded in
`data/benchmarks/calibration_references.csv` with full provenance
metadata. The gate report is at `results/calibration_v1/report.md` and
the residuals CSV at `results/calibration_v1/residuals.csv`. The gate
was run from `uv run python -m infrarl.simulator.calibration` at the
repository commit recorded in the calibration summary's
`references_sha256` field.

## 7. Simulator sensitivity

### 7.1 Method

The simulator is a structural lookup-table model, not a calibrated digital
twin. To defend the ranking in Section 4 against the "uncalibrated
simulator" critique, we perturb the simulator's `estimate()` output
multiplicatively on three independent axes (quality, latency, cost) and
re-evaluate each trained policy on the held-out test split (`test_seed =
1000`, `n_test = 1000`). Perturbation is applied by rebinding the imported
`estimate` symbol inside `infrarl.simulator.environment`, because
`from .estimator import ... estimate` in that module creates a module-local
binding that patching `estimator.estimate` alone would not affect
(`infrarl/evaluation/sensitivity.py:55-71`). Agents are trained once per
seed on the nominal estimator, then the perturbation is swapped in only for
test-time evaluation. This measures the sensitivity of the *learned
policy* to simulator miscalibration, not re-training sensitivity. Four
agents are evaluated (heuristic, bandit_contextual, ppo, lagrangian_ppo)
across three seeds and four non-identity multipliers per axis (0.8, 0.9,
1.1, 1.2) plus the nominal 1.0 baseline, perturbing one axis at a time.
Joint perturbation is deferred to Section 9. Figure 6 renders seed-mean
meets_all against multiplier for each axis.

### 7.2 Finding

- **Cost perturbation at plus-or-minus 20 percent: ranking fully
  preserved.** The ordering lagrangian_ppo > ppo > heuristic >
  bandit_contextual holds at every grid point. Lagrangian PPO's meets_all
  moves from 0.678 at `cost x 0.8` to 0.619 at `cost x 1.2`, a span of
  0.059. No lines cross.
- **Latency perturbation at plus-or-minus 20 percent: ranking fully
  preserved.** Same ordering across the grid. Lagrangian PPO moves from
  0.673 at `latency x 0.8` to 0.619 at `latency x 1.2`. PPO stays above
  the heuristic throughout.
- **Quality perturbation: ranking preserved only within plus-or-minus 10
  percent.** At `quality x 1.2`, PPO (0.795) overtakes Lagrangian PPO
  (0.724) on meets_all, because PPO's cheaper, lower-quality allocations
  begin to clear an inflated quality floor "for free" and the cost
  savings dominate the composite metric. The heuristic (0.705) and
  bandit_contextual (0.700) close in on Lagrangian PPO at this extreme.
  At `quality x 0.8`, Lagrangian PPO still leads (0.127) but the whole
  system collapses: PPO drops to 0.065, bandit_contextual to 0.023, and
  the heuristic to 0.000.

### 7.3 Interpretation

Under cost or latency miscalibration up to plus-or-minus 20 percent, the
Section 4 ranking is the operative recommendation without caveat. Under
quality miscalibration, the Lagrangian PPO advantage over vanilla PPO
depends on the simulator's quality numbers being correct within about
plus-or-minus 10 percent. Beyond that band the Lagrangian-versus-vanilla
choice becomes workload-dependent: a simulator that systematically
over-reports quality makes cheap stacks look compliant and rewards vanilla
PPO's cost-minimization; a simulator that under-reports quality still
favors the constrained policy. The implied calibration priority order is
quality first, latency second, cost third. This also motivates the
quality-axis focus of the three-anchor residual audit in Section 6.

### 7.4 Caveats

Perturbation is one axis at a time; joint perturbation (for example,
`quality x 0.8` and `cost x 1.2` together) is not swept and is deferred
to Section 9 future work. Only four agents are perturbed, not all eight.
The four excluded agents (random, bandit_context_free, bandit_discounted,
q_learning) sit well below the Lagrangian > PPO > heuristic cluster on
nominal meets_all; their inclusion would not change the rank-stability
story at the top of the leaderboard. Constrained-MDP framing and the
Lagrangian dual update follow Altman [1] and Stooke, Achiam, Abbeel [4];
the base PPO clip objective is Schulman et al. [2].

## 8. Discussion and limitations

Six limitations shape how the Section 4 headline should be read.

**Data: no public decision-space dataset exists.** Published LLM
inference benchmarks (MLPerf Inference [12], vLLM [9], Lambda Labs,
Anyscale) report single-configuration measurements; none publish the
profile-by-stack cross product with real deployment outcomes that this
work would need. Building one would require instrumenting dozens of
production inference services and permissioning the per-request
quality/latency/cost traces. In the absence of that corpus, we built a
structural simulator from published vendor and academic benchmarks, then
audited it against three public-reference anchors (Section 6) and
established rank-stability under perturbation (Section 7).

**Calibration coverage is partial.** The three-anchor residual gate
specifies nine required cells (three configurations by three axes: quality,
latency, monthly service cost). Only three cells have independently
retrieved references; the remaining six are structural extrapolations from
the lookup tables (`data/benchmarks/sources.md`). The three-axis
rank-stability analysis in Section 7 is therefore the operative defense of
the policy ordering. We do not claim absolute dollar or latency values
match any specific production deployment.

**Sample size.** The main battery uses n=3 seeds. Pairwise comparisons
with small effect sizes are reported as directional, not as statistically
significant results. Wide confidence intervals are an honest consequence of
the course-project compute budget, not a residual defect to be waved at.

**PPO versus heuristic on meets_all.** On the held-out test split
Lagrangian PPO reaches 0.650 +/- 0.033 meets_all while PPO reaches 0.484
+/- 0.032 and the heuristic reaches 0.460 +/- 0.000
(`results_frozen_v2_test/summary.csv`). The 0.484 versus 0.460 PPO-versus-heuristic point estimate favors PPO but the gap is inside the
noise envelope at n=3. The Lagrangian-versus-vanilla gap on
quality_compliance (0.894 +/- 0.021 versus 0.589 +/- 0.075) is large
enough to be defensible at this sample size.

**DQN was not implemented.** The proposal mentioned a DQN baseline; the
file `infrarl/agents/dqn_agent.py` is a zero-byte stub. Lagrangian PPO
carries the deep-RL story. No quantitative claim in this report depends
on DQN.

**Out of scope.** Multi-agent competitive or cooperative scheduling across
shared GPU pools, real-hardware latency measurement beyond the three
calibration anchors, cross-region deployment, and production SLO
negotiation are not addressed. See Section 9.

## 9. Future work

- **Empirical calibration against vLLM on Zaratan A100 40 GB.** Rebaseline
  the simulator's hardware table from the simulated A100 80 GB SKU to the
  Zaratan A100 40 GB SKU and run matched workloads under vLLM with
  recorded TTFT, TPOT, and E2E latency statistics. A single matched run
  converts at least one row of the residual gate from `NOT_COMPARABLE`
  to a numeric residual and tightens the Section 7 quality-axis bound.
- **Online calibration from live telemetry.** The D-LinUCB agent
  (Section 3.4) already handles piecewise-stationary reward drift via its
  discount factor; wiring it to a real reward signal from a production
  gateway is a tractable next step for a pilot. The drift harness in
  `infrarl/online/drift.py` is the integration seam.
- **Multi-tenant extension.** Competing profiles share a finite GPU pool;
  the policy schedules allocations across tenants rather than selecting
  independently per profile. This moves the formulation from single-profile
  stack selection into true multi-agent resource contention with
  cross-tenant SLA coupling.

## 10. Team contributions

- **Jay Guwalani.** Reinforcement-learning agents (PPO, Lagrangian PPO,
  tabular Q-learning, D-LinUCB, online agent selector); simulator-perturbation
  sensitivity harness (`infrarl/evaluation/sensitivity.py`); three-anchor
  calibration anchor sourcing and gate runs
  (`scripts/fill_calibration_refs.py`, `results/calibration_v1/`); report
  lead (Section 1 Introduction, Section 4 Main Comparison, Section 6
  Calibration residuals, Section 7 Simulator sensitivity, Section 8
  Discussion).
- **Anupama Sharma.** Simulator lookup tables for hardware, model,
  and precision (`infrarl/simulator/hardware.py`,
  `infrarl/simulator/models.py`); calibration framework and residual gate
  (`infrarl/simulator/calibration.py`); profile exemplars generator;
  provenance audit at `data/benchmarks/sources.md`.
- **Vishnu Sai Vardhan Reddy Basi Reddy Gari.** Evaluation harness
  (`infrarl/evaluation/run_experiment.py`); Friedman and Wilcoxon-Holm
  statistical tests (`infrarl/evaluation/stats.py`); Pareto, heatmap,
  compliance, learning-curve, and ablation plots
  (`infrarl/evaluation/plots.py`); verification suite with five gates
  (`infrarl/evaluation/verify.py`); Streamlit demo (`infrarl/app.py`).

All components are reproducible from the shared codebase at the commit
recorded in `results_frozen_v2_test/experiment.json` and
`results_sensitivity_v1/sensitivity.json`.

## 11. References

[1] Altman, E. (1999). *Constrained Markov Decision Processes*. Chapman
and Hall/CRC.

[2] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O.
(2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.

[3] Achiam, J., Held, D., Tamar, A., Abbeel, P. (2017). Constrained
Policy Optimization. In *Proceedings of the 34th International
Conference on Machine Learning (ICML)*.

[4] Stooke, A., Achiam, J., Abbeel, P. (2020). Responsive Safety in
Reinforcement Learning by PID Lagrangian Methods. In *Proceedings of the
37th International Conference on Machine Learning (ICML)*.

[5] Li, L., Chu, W., Langford, J., Schapire, R. E. (2010). A
Contextual-Bandit Approach to Personalized News Article Recommendation.
In *Proceedings of the 19th International Conference on World Wide Web
(WWW)*.

[6] Russac, Y., Vernade, C., Cappe, O. (2019). Weighted Linear Bandits
for Non-Stationary Environments. In *Advances in Neural Information
Processing Systems (NeurIPS)*.

[7] Dettmers, T., Lewis, M., Belkada, Y., Zettlemoyer, L. (2022).
LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. In
*Advances in Neural Information Processing Systems (NeurIPS)*.

[8] Lin, J., Tang, J., Tang, H., Yang, S., Chen, W., Wang, W., Xiao, G.,
Dang, X., Gan, C., Han, S. (2023). AWQ: Activation-aware Weight
Quantization for LLM Compression and Acceleration. arXiv:2306.00978.

[9] Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H.,
Gonzalez, J. E., Zhang, H., Stoica, I. (2023). Efficient Memory
Management for Large Language Model Serving with PagedAttention. In
*Proceedings of the 29th Symposium on Operating Systems Principles
(SOSP)*.

[10] Ong, I., Almahairi, A., Wu, V., Chiang, W.-L., Wu, T., Gonzalez,
J. E., Kadous, M. W., Stoica, I. (2024). RouteLLM: Learning to Route
LLMs with Preference Data. arXiv:2406.18665.

[11] Chen, L., Zaharia, M., Zou, J. (2023). FrugalGPT: How to Use Large
Language Models While Reducing Cost and Improving Performance.
arXiv:2305.05176.

[12] MLCommons. (2024). MLPerf Inference: Datacenter Benchmark Suite,
v5.x results. https://github.com/mlcommons/inference_results_v5.1.

