# Evaluation harness (Vishnu)

One command trains every registered method, evaluates it greedily on a held-out set,
and writes tables, statistics and figures.

```bash
python -m infrarl.evaluation.run_experiment --agents random,heuristic,bandit_context_free,bandit_contextual,ppo --seeds 3 --out results/NEW_DEVELOPMENT_RUN
# DQN and Q-learning are placeholders; do not request them until implemented.
streamlit run infrarl/app.py
```

## Protocol
- Train profiles: 1000 synthetic profiles, regenerated per seed (`seed=2000+s`).
- Validation profiles: seed 3000 for curves and selection.
- Final test profiles: 1000 fixed held-out profiles (`seed=1000`), identical for all methods, evaluated only after frozen training.
- Learning agents train for `train_epochs` passes (shuffled), then one greedy pass with no updates.
- Statistics: Friedman across methods per seed, then pairwise Wilcoxon signed-rank with Holm correction; mean difference reported as effect size, alpha 0.05.
- Heuristic and random are not learning agents; the heuristic is deterministic, so its std over seeds is 0.

## Outputs (`results/`, git-ignored)
`episodes.csv` per-profile outcomes, `summary_per_seed.csv`, `summary.csv` (mean/std), `curves.csv`, `stats.csv`,
`ablations.csv`, and `figures/` (Pareto, per-use-case heatmap, compliance bars, learning curves, ablations).

## Metrics
`avg_quality` is raw simulated quality, with no global numeric target. Because required accuracy is drawn from 0.60-0.90, the average requirement is about 0.75. Use `quality_compliance` (quality >= the profile's requirement) and `quality_attainment` (quality / requirement, clipped at 1). `meets_all` = quality, budget and SLA all satisfied.

## Plugging in a new agent (Jay: Q-learning, DQN, PPO)
1. Implement `agents/base.py::Agent`: `begin_episode(profile)`, `select_action(obs, mask, explore)` returning a legal action, `update(transition)` called after every env step (reward is 0 until the terminal step).
2. Define the class as `QLearningAgent`, `DQNAgent` or `PPOAgent` in its file with `__init__(self, seed=0)`; the registry auto-registers it (30 epochs). Or call `register(name, factory, train_epochs)` yourself.
3. Run the command above with `--agents <name>`. Ablations work for any mask-aware agent with no changes.

## Ablations
`no_hardware_awareness` (A100 only), `no_precision_tuning` (fp16, falling back only when it does not fit), `no_scaling_policy` (static), `no_multitier`. They restrict the env's action mask. Bandits play whole single-tier arms, so they also get an arm filter, and `no_multitier` is a no-op for them. Restricting the action space also shortens learning, so run ablations on a converged agent before drawing conclusions.

## Plugging in a better simulator (Anupama)
Everything goes through `simulator/estimator.py::estimate(stack, profile) -> (quality, latency_ms, cost)`; replace the tables in `hardware.py` and `models.py` and the harness is unchanged. `tests/test_simulator.py` pins the smoke-test numbers.

## Verification
```bash
python -m infrarl.evaluation.verify --out results/verification.json
pytest -q tests/test_verification.py
```
Exhaustive mask enumeration (1- and 2-tier stacks; path counts must equal T^k * 3 where T = memory-feasible tiers), simulator property checks, reward boundary cases, determinism, and a brute-force oracle gap. `tests/test_verification.py` also injects a mask bug and a simulator bug and asserts the suite catches them. The historical H100 price assertion is a hardcoded internal regression check, not a newly retrieved or source-matched calibration record. Current external price/model/workload coverage is documented in data/benchmarks/sources.md.

## Safe Zaratan workflow and validation split

Use the shared phase runner documented in [the Zaratan workflow](../scripts/zaratan/README.md).
`--agents` is now required, output directories must be new, and ablations default
to disabled. Ordinary driver invocation evaluates **validation**, not final test:

```bash
python -m infrarl.evaluation.run_experiment --agents random,heuristic,bandit_context_free,bandit_contextual,ppo --evaluation-split validation --out results/NEW_DEVELOPMENT_RUN
```

Training profiles use seed `2000+s`; validation profiles use **3000** for epoch
curves and model selection. `--evaluation-split test` evaluates seed **1000** only
after training and must be reserved for scientifically approved frozen reporting.
Curves never use test profiles. Freeze the inventory, simulator and hyperparameters
before examining test results. Empty Q-learning/DQN modules are not available agents.

Ablations restrict legal actions; budgets match seeds, profile exposures and epochs,
not necessarily transitions or optimizer steps. `no_precision_tuning` permits
INT8/INT4 fallback when FP16 does not fit. Functional smoke and PPO optimizer-update
smoke are separate gates. Calibration coverage is independent of correctness tests;
missing matched references require provisional labels and explicit authorization.

## Calibration, exemplars and CPU execution

The canonical profile requirements are `acc_req ~ Uniform(0.60,0.90)`. Reward
weights are `(0.5,0.3,0.2)` with the 0.3 quality-compliance bonus inside `q_r`;
maximum terminal reward is 1.18. These settings are unchanged.

Use fresh output directories; calibration reports incomplete coverage explicitly:

```bash
python scripts/run_calibration.py --out results/NEW_CALIBRATION_RUN
python scripts/generate_profile_exemplars.py --out results/NEW_EXEMPLAR_RUN
```

Calibration exits 0 for complete required matched coverage, 1 for a confirmed
failed/infeasible gate, and 2 for incomplete evidence. Independent references and
holdouts remain missing; label research outputs `PROVISIONAL_UNCALIBRATED`.
See [source provenance](../data/benchmarks/sources.md) for the coverage matrix.
Exemplars are six actual generated synthetic rows, selected reproducibly from
1000 profiles with seed 1000; no trained-policy recommendations are invented.

Run locally with an existing compatible environment and limited CPU threads, or
use the shared CPU Slurm entry point in [scripts/zaratan/README.md](../scripts/zaratan/README.md).
All cluster computation requires an allocated compute node. Package versions,
source hashes, seeds and per-run configurations belong with the result artifacts.
