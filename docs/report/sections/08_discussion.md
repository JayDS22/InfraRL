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
