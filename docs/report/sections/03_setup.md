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
