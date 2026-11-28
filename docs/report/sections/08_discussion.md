## 8. Discussion and limitations

### 8.1 Primary finding: sensitivity and calibration together are necessary and sufficient

The single most important result in this paper is not the Table 2A or
Table 2B leaderboard. It is the methodological observation that neither
a sensitivity analysis nor a calibration audit is, by itself, sufficient
to establish whether a structural simulator supports a production
recommendation.

Our Section 7 sensitivity sweep established that the agent ranking is
stable under multiplicative perturbation of the simulator outputs within
plus-or-minus 20 percent per axis. That result was correct and
reproducible but tells the reader nothing about whether the production
reality actually sits inside that envelope. Our Section 6 calibration
audits (three in-depth anchors plus 29 broad references) answered that
question: real single-stream decode throughput sits 8 to 31 times outside
the sensitivity envelope on the latency axis. Section 4.3 showed that
when the real miscalibration is applied, the policy ranking flips
exactly as Section 7 would predict for perturbations beyond the tested
envelope.

Two takeaways generalize beyond this project. First, a sensitivity
envelope is a cheap necessary condition for simulator validity (if
the ranking is unstable even to small perturbations, the simulator
cannot support any policy recommendation); it is not a sufficient
condition. Second, calibration tells you whether reality is inside the
envelope. We ran both; our ranking flipped at exactly the point the
sensitivity analysis said it would flip. That empirical confirmation of
a methodological hypothesis is the research contribution of this work.

### 8.2 Limitations

Seven caveats shape how Sections 4, 6, and 7 should be read.

**Data: no public decision-space dataset exists.** Published LLM
inference benchmarks (MLPerf Inference [12], vLLM [9], Lambda Labs,
Anyscale) report single-configuration measurements; none publish the
profile-by-stack cross product with real deployment outcomes that this
work would need. Building one would require instrumenting dozens of
production inference services and permissioning the per-request
quality/latency/cost traces. In the absence of that corpus, we built a
structural simulator from published vendor and academic benchmarks and
calibrated its primitive constants against measured points (Section 6).

**Calibration coverage is uneven across use cases.** The 29-reference
recalibration covers chatbot and code_generation quality (sourced from
Qwen2.5 benchmarks) but not RAG, agentic, batch_inference, or
structured_extraction; no matched public benchmark exists at the
simulator's use-case granularity for those four. Those quality cells
retain pre-recalibration values and should be read as structural
extrapolations. A10 and CPU decode speed have no public measured
reference; they scaled from the A100 fit and carry an UNVERIFIED flag
in the hardware table.

**Sample size.** The main battery uses n = 3 seeds. Within-seed
Wilcoxon tests are powered by n = 1000 paired profiles and give stable
per-seed verdicts. Across-seed std bands have n = 3 and should be read
as directional, not as narrow confidence intervals. Pairwise
comparisons with small effect sizes (bandit versus bandit at the
leaderboard tail) are reported as directional; large effects
(Lagrangian PPO versus PPO on quality_compliance at 17 sigma
pre-recalibration) are defensible.

**Profile SLAs may be unrealistically tight.** Section 4.3 showed that
only 34.5 percent of generated profiles are satisfiable by any single-
or two-tier stack under the recalibrated simulator (vs 85.8 percent
pre-recalibration). Many of the synthetic customer SLAs (e.g. a
chatbot requiring a 200-token completion in 400-1200 ms) are not
achievable at measured single-stream decode speeds. Section 9 lists a
realistic-SLA generator as the natural follow-up; its omission here is
why the absolute meets_all values in Table 2B are small.

**Ablations run only on Lagrangian PPO (pre-recalibration).** The
Section 4.5 ablation table isolates the environmental effect of
restricting the action space from the agent effect of how Lagrangian
PPO reacts to the restriction. A vanilla-PPO ablation (available in
`results/final_20261125/ablate_ppo/` from PR #4) is not integrated
into the main narrative. The confounding is flagged in Section 9.

**DQN was not implemented.** The proposal mentioned a DQN baseline;
the file `infrarl/agents/dqn_agent.py` is a zero-byte stub. Lagrangian
PPO carries the deep-RL story. No quantitative claim in this report
depends on DQN.

**Out of scope.** Multi-agent competitive or cooperative scheduling
across shared GPU pools, real-hardware latency measurement beyond
the 29 calibration references, cross-region deployment, production
SLO negotiation, and online retraining under drift beyond the
Section 5 four-week horizon are not addressed. See Section 9.
