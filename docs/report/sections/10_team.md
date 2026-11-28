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
