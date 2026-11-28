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
