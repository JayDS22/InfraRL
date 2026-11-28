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

