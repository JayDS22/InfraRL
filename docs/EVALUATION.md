# Evaluation harness (Vishnu)

One command trains every registered method, evaluates it greedily on a held-out set,
and writes tables, statistics and figures.

```bash
python -m infrarl.evaluation.run_experiment --seeds 3 --out results
python -m infrarl.evaluation.run_experiment --agents dqn,heuristic,random --ablate dqn --epochs 30
streamlit run infrarl/app.py
```

## Protocol
- Train profiles: 1000 synthetic profiles, regenerated per seed (`seed=2000+s`).
- Test profiles: 1000 fixed held-out profiles (`seed=1000`), identical for all methods, so tests are paired.
- Learning agents train for `train_epochs` passes (shuffled), then one greedy pass with no updates.
- Statistics: Friedman across methods per seed, then pairwise Wilcoxon signed-rank with Holm correction; mean difference reported as effect size, alpha 0.05.
- Heuristic and random are not learning agents; the heuristic is deterministic, so its std over seeds is 0.

## Outputs (`results/`, git-ignored)
`episodes.csv` per-profile outcomes, `summary_per_seed.csv`, `summary.csv` (mean/std), `curves.csv`, `stats.csv`,
`ablations.csv`, and `figures/` (Pareto, per-use-case heatmap, compliance bars, learning curves, ablations).

## Metrics
`avg_quality` is the proposal metric (raw quality, target 0.85). Because required accuracy is drawn from 0.60-0.90, the average requirement is about 0.75, so a cost-aware agent cannot reach 0.85 on average. Prefer `quality_compliance` (quality >= the profile's requirement) and `quality_attainment` (quality / requirement, clipped at 1). `meets_all` = quality, budget and SLA all satisfied.

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
Exhaustive mask enumeration (1- and 2-tier stacks; path counts must equal T^k * 3 where T = memory-feasible tiers), simulator property checks, reward boundary cases, determinism, and a brute-force oracle gap. `tests/test_verification.py` also injects a mask bug and a simulator bug and asserts the suite catches them. Price anchor: Nebius lists HGX H100 at $4.50/GPU-hour (nebius.com/prices); it lists no A100 or A10, so those two prices have no Nebius anchor yet.
