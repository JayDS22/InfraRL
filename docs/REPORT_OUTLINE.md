# InfraRL Final Report: Outline and Delivery Plan

| Field | Value |
|---|---|
| Document ID | INFRARL-DOC-001 |
| Version | 0.1 (draft) |
| Status | DRAFT |
| Course | DATA635, Reinforcement Learning, UMD, Fall 2025 |
| Instructor | Prof. Samet Ayan |
| Report due | 2026-12-07 |
| Document owner | Jay Guwalani |
| Contributors | Anupama (simulator, calibration), Vishnu (evaluation, demo) |
| Last updated | 2026-11-25 |
| Source repository | `github.com/JayDS22/InfraRL` |

## 1. Purpose

This document defines the structure, deliverables, and acceptance criteria
for the InfraRL final report. It is the working source of truth for section
scope, figure and table inventory, dependencies on in-flight experiments,
and the writing schedule up to the 2026-12-07 submission.

The final report itself will be rendered from this outline into LaTeX in the
final delivery week. This document does not constrain prose style, only the
structural contract.

## 2. Audience

- Primary: Prof. Samet Ayan, grading the DATA635 final deliverable.
- Secondary: future DATA635 students or external readers landing on the
  repository; the report doubles as the project's canonical writeup.

## 3. Scope

### 3.1 In scope

- Problem formulation: LLM inference infrastructure selection as a
  constrained Markov decision process.
- Eight-agent empirical comparison on a held-out test split.
- Lagrangian PPO constrained-RL contribution and ablations.
- Online adaptation to reward-weight drift.
- Simulator-perturbation sensitivity analysis.
- Three-anchor residual calibration audit against public references.

### 3.2 Out of scope

- Multi-tenant scheduling across competing profiles.
- Real-hardware latency measurement beyond the three calibration anchors.
- Cross-cluster or multi-region deployment.
- DQN agent (stub only, cut from final narrative).
- Production deployment, SLO negotiation, cost reclamation.

## 4. Acronyms and defined terms

| Term | Definition |
|---|---|
| CMDP | Constrained Markov Decision Process |
| GAE | Generalized Advantage Estimation |
| MDP | Markov Decision Process |
| PPO | Proximal Policy Optimization (Schulman et al. 2017) |
| LinUCB | Linear Upper Confidence Bound bandit (Li et al. 2010) |
| D-LinUCB | Discounted LinUCB (Russac, Vernade, Cappe 2019) |
| SLA | Service Level Agreement, request latency budget |
| Compliance | Fraction of profiles meeting a given constraint |
| Meets-all | Compliance on all three axes (quality, budget, SLA) simultaneously |
| Residual gate | Percent relative error between simulator prediction and reference |
| Anchor | A `(hardware, model, precision)` reference triple used by the gate |
| Rank stability | Preservation of agent ordering across perturbation grid |

## 5. Report structure

Section numbering below mirrors the final report. Each section specifies
purpose, target length, inputs (artifacts required before writing), and
deliverables (what the section produces for the reader).

### 5.1 Abstract

- **Purpose.** Single paragraph summarizing problem, method, headline
  result, and principal limitation.
- **Target length.** 150 words, 1 paragraph.
- **Inputs.** Final results from Sections 5.5 and 5.7.
- **Deliverables.** Abstract paragraph suitable for table-of-contents
  previews.

### 5.2 Introduction

- **Purpose.** Establish the problem and motivate learning over static
  heuristics. State contributions in order of defensibility.
- **Target length.** 1 page.
- **Inputs.** Project proposal (`proposal.md`), Ayan-review notes.
- **Deliverables.**
  1. Problem statement: given a request profile, select an inference
     stack (hardware, model, precision, scaling) subject to quality,
     budget, and SLA constraints.
  2. Motivation for learning: profile distribution shift, nonlinear
     trade-offs, context-aware selection.
  3. Motivation for constrained MDP: quality is a hard floor, not an
     average.
  4. Four-item contribution list: Lagrangian PPO policy, drift
     adaptation, sensitivity analysis, residual calibration audit.
  5. Explicit non-claim: this is not a calibrated digital twin.

### 5.3 Related work

- **Purpose.** Place the contribution in the landscape; identify the gap.
- **Target length.** Half page.
- **Inputs.** Reference list (Section 5.11).
- **Deliverables.** Three threads (model routing, RL for cluster
  scheduling, constrained MDPs) and one sentence identifying the gap
  this work closes.

### 5.4 Setup

- **Purpose.** Define problem formulation, simulator, environment,
  agents, and evaluation protocol with sufficient detail to reproduce.
- **Target length.** 2 pages.
- **Inputs.** Code at commit 6deb339.
- **Subsections.**

#### 5.4.1 Problem formulation

Profile features, action space, reward function with bounds and weights
per `infrarl/simulator/environment.py:48`.

#### 5.4.2 Simulator

Three lookup tables and the `estimate(stack, profile)` function.
Explicit statement that this is a structural model, not a calibrated
simulator; forward-reference to Section 5.7.

#### 5.4.3 Environment and action mask

Gymnasium wrapper, per-step mask, verification suite in
`infrarl/evaluation/verify.py` enforcing mask exhaustiveness, simulator
monotonicity, reward bounds, determinism, and oracle gap. All five
gates passing at the reference commit.

#### 5.4.4 Agents

Table 1 (see Section 5.9) enumerates all eight agents with references.

#### 5.4.5 Evaluation protocol

Disjoint seed regime (train 2000+s, validation 3000, test 1000), 3
seeds, Friedman + Wilcoxon-Holm per seed, headline metric ordering.
Explicit n=3 caveat.

### 5.5 Main comparison

- **Purpose.** Report the frozen-battery headline on the held-out test
  split.
- **Target length.** 2 pages.
- **Inputs.** `results_frozen_v2_test/` from Zaratan job 24276167.
- **Deliverables.**
  - Table 2: eight agents by seven headline metrics, mean plus/minus
    std over 3 seeds.
  - Figure 1: Pareto plot with Pareto front overlay.
  - Figure 2: compliance bar chart grouped by agent.
  - Figure 3: Lagrangian PPO ablation deltas across four variants.
  - Figure 4: learning curves.
  - Statistical significance row: Wilcoxon-Holm p-values for
    Lagrangian PPO vs PPO and PPO vs heuristic.
  - Two-paragraph narrative framing the quality-compliance versus cost
    trade-off; Lagrangian PPO wins compliance at 2.07x PPO cost.

### 5.6 Drift adaptation

- **Purpose.** Demonstrate that online agents recover from reward-weight
  shifts that defeat a frozen policy.
- **Target length.** 1 page.
- **Inputs.** `results_frozen_v2_test/drift/drift.csv` and `drift.png`.
- **Deliverables.**
  - Experimental setup paragraph: reward weights shift mid-stream.
  - Figure 5: weekly reward trajectory, three lines (static Lagrangian
    PPO, D-LinUCB, online selector), drift event marked with vertical
    line.
  - Finding statement: online agents recover within approximately two
    weeks; the frozen policy remains underwater for the remainder of
    the horizon.

### 5.7 Simulator sensitivity

- **Purpose.** Establish that agent ranking is stable to simulator
  miscalibration within plus or minus 20 percent on each axis.
- **Target length.** 1 page.
- **Inputs.** `results_sensitivity_v1/sensitivity.csv` from Zaratan job
  24276168, rendered via `infrarl.evaluation.plots.plot_sensitivity`.
- **Deliverables.**
  - Method paragraph: multiplicative perturbation of the estimator
    output, one axis at a time, re-evaluation on held-out test.
  - Figure 6: three-panel sensitivity plot, meets-all on y, multiplier
    on x, one line per agent.
  - Rank-stability finding: pairs of agents that cross versus pairs
    that remain separated.
  - Operative claim: if the real system diverges from the simulator by
    less than 20 percent on any single axis, our conclusions about
    policy ranking do not flip.

### 5.8 Calibration residuals

- **Purpose.** Report the three-anchor residual gate audit against
  public references.
- **Target length.** 1 page.
- **Inputs.** `results/calibration_v1/summary.json` and `report.md`
  after Anupama fills `data/benchmarks/calibration_references.csv`.
- **Deliverables.**
  - Table 3: nine required cells by configuration and axis, with
    prediction, reference, residual percent, and gate status.
  - Honest outcome statement: residuals exceed tolerance on latency
    axis (simulator throughput is optimistic); quality axis is not
    directly comparable to published MMLU or HumanEval scores; cost
    axis passes within tolerance for at least one anchor.
  - Reframing: this is not a calibrated digital twin; claims about
    relative policy ordering remain defensible via Section 5.7.

### 5.9 Tables and figures inventory

| ID | Caption | Source artifact | Status |
|---|---|---|---|
| Table 1 | Agent inventory with algorithm reference | (hand-coded) | pending |
| Table 2 | Main comparison across eight agents | `results_frozen_v2_test/summary.csv` | blocked on job 24276167 |
| Table 3 | Calibration residual summary | `results/calibration_v1/residuals.csv` | blocked on anchor fill |
| Figure 1 | Pareto: quality vs cost | `figures/pareto_quality_vs_cost.png` | blocked on job 24276167 |
| Figure 2 | Compliance bar chart | `figures/compliance.png` | blocked on job 24276167 |
| Figure 3 | Ablation deltas | `figures/ablations.png` | blocked on job 24276167 |
| Figure 4 | Learning curves | `figures/learning_curves.png` | blocked on job 24276167 |
| Figure 5 | Drift trajectory | `drift/drift.png` | blocked on job 24276167 |
| Figure 6 | Sensitivity rank stability | `sensitivity.png` | blocked on job 24276168 |

### 5.10 Discussion and limitations

- **Purpose.** State limitations explicitly rather than relying on the
  reader to infer them.
- **Target length.** Half page.
- **Deliverables.**
  1. Calibration gate is partial; only three of nine required cells
     have references. Section 5.7 is the operative defense.
  2. Sample size: n=3 seeds. Small effect sizes are reported as
     directional, not statistically significant.
  3. Vanilla PPO does not reject the heuristic baseline on meets-all
     at n=3; Lagrangian PPO closes this gap by making the quality
     constraint explicit.
  4. DQN agent was not implemented; `infrarl/agents/dqn_agent.py` is
     a stub. Lagrangian PPO carries the deep-RL story.
  5. No multi-agent interaction, no real-hardware latency beyond the
     three anchors, no production validation. See Section 5.11.

### 5.11 Future work

- **Purpose.** Convert known limitations into concrete follow-ups.
- **Target length.** Quarter page.
- **Deliverables.** Three bullets:
  1. Empirical calibration against vLLM on Zaratan A100 40GB with the
     simulator table rebaselined to the 40GB SKU.
  2. Online calibration from live telemetry; D-LinUCB already supports
     non-stationarity.
  3. Multi-tenant extension: shared GPU pool across competing profiles,
     policy schedules allocations.

### 5.12 Team contributions

- **Purpose.** Document per-contributor responsibility consistent with
  course policy on group work.
- **Target length.** Quarter page, 3 bullets.
- **Deliverables.** One sentence per contributor listing concrete
  components owned.

### 5.13 References

- **Purpose.** IEEE-style numbered list of citations used in the body.
- **Target length.** As required.
- **Deliverables.** Minimum 10 references:

| # | Citation | Appears in |
|---|---|---|
| 1 | Altman, E. (1999). Constrained Markov Decision Processes. | Sections 5.2, 5.4 |
| 2 | Schulman, J. et al. (2017). Proximal Policy Optimization. arXiv:1707.06347. | Section 5.4 |
| 3 | Achiam, J. et al. (2017). Constrained Policy Optimization. ICML. | Section 5.3 |
| 4 | Stooke, A., Achiam, J., Abbeel, P. (2020). Responsive Safety in RL. ICML. | Section 5.4 |
| 5 | Li, L. et al. (2010). A Contextual-Bandit Approach to News Article Recommendation. WWW. | Section 5.4 |
| 6 | Russac, Y., Vernade, C., Cappe, O. (2019). Weighted Linear Bandits for Non-Stationary Environments. NeurIPS. | Section 5.4 |
| 7 | Dettmers, T. et al. (2022). LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. NeurIPS. | Section 5.8 |
| 8 | Lin, J. et al. (2023). AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. arXiv:2306.00978. | Section 5.8 |
| 9 | Kwon, W. et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM). SOSP. | Section 5.8 |
| 10 | Ong, I. et al. (2024). RouteLLM: Learning to Route LLMs with Preference Data. arXiv:2406.18665. | Section 5.3 |

Add FrugalGPT (Chen et al. 2023) and MLPerf Inference v5.x if Section 5.3
references them in prose.

## 6. Dependencies and critical path

| Dependency | Owner | Target date | Status | Blocks |
|---|---|---|---|---|
| Zaratan job 24276167 (frozen v2 test split) | Jay | 2026-11-26 | scavenger queue, PD | Sections 5.5, 5.6 |
| Zaratan job 24276168 (sensitivity sweep) | Jay | 2026-11-26 | scavenger queue, PD | Section 5.7 |
| Anchor fill in `calibration_references.csv` | Anupama | 2026-11-30 | not started | Section 5.8 |
| Local sensitivity figure rendering | Jay | after 24276168 completes | not started | Figure 6 |
| LaTeX template selection | Jay | 2026-12-03 | not started | Final formatting |

Critical path: Zaratan jobs must complete and sync back before Section 5.5,
5.6, 5.7 can be written. Anchor fill gates Section 5.8. LaTeX conversion
gates final submission.

## 7. Deliverable schedule

| Milestone | Target | Contents |
|---|---|---|
| M1: Outline frozen | 2026-11-25 | This document, v0.1 |
| M2: Sections 1-4 draft | 2026-11-28 | Non-results sections (abstract, intro, related, setup) |
| M3: Results in hand | 2026-11-27 (slip tolerance +2 days) | All Zaratan artifacts synced to Mac, figures rendered |
| M4: Sections 5-8 draft | 2026-12-01 | Results and discussion sections |
| M5: Full draft | 2026-12-03 | Markdown complete, peer read by Anu and Vishnu |
| M6: LaTeX conversion | 2026-12-05 | Compiling PDF, figures at final size |
| M7: Submission | 2026-12-07 | PDF submitted per course instructions |

## 8. Open questions

1. LaTeX template: IEEE two-column or ACM single-column acmtog? Confirm
   course requirement.
2. Figure 6 (sensitivity): one metric (meets_all) or two (meets_all and
   quality_compliance)? Default to one; revisit if the story needs both.
3. Supplementary materials: include `results_frozen_v2_test/stats.csv`
   and `results_sensitivity_v1/sensitivity.csv` as an appendix or link
   the repo commit only?
4. Author ordering: alphabetical or by contribution weight? Default to
   alphabetical by first name for the DATA635 cover sheet.

## 9. Change log

| Version | Date | Author | Changes |
|---|---|---|---|
| 0.1 | 2026-11-25 | Jay Guwalani | Initial outline. Structure, dependencies, references, and schedule drafted. All results sections marked blocked on Zaratan jobs 24276167 and 24276168. |
