# InfraRL live demo

A one-page web demo that runs the **real trained agents** through the InfraRL simulator.
Nothing on screen is precomputed: each click runs the saved agents step by step and scores
their stack with the simulator.

```bash
# from the repo root, using the project .venv (needs starlette + uvicorn, already installed)
python -m webdemo.pretrain --sim placeholder   # only if webdemo/models/ is empty
python -m webdemo.pretrain --sim calibrated
python -m webdemo.server       # then open http://localhost:8501
```

## Two screens
1. **Recommend for one customer**: pick a preset (startup chatbot, code assistant, batch job,
   RAG, agent platform), a random held-out test customer, or set the sliders. Each agent's 5
   decisions (architecture, hardware, model, precision, scaling) appear step by step, then the
   accuracy / latency / budget checks. The last lane is an exhaustive search over all 7,956
   one- and two-tier setups, as a yardstick.
2. **Head-to-head on unseen customers**: all agents on 25-200 customers sampled from the same
   1,000 held-out test profiles (seed 1000) used in the evaluation; the scoreboard fills live.

## Demo models
`webdemo/models/*.pkl` are trained with the evaluation recipe (seed 0, 1000 training profiles,
registry epoch budget). On the full test set they score meets-all: Lagrangian PPO 0.657,
heuristic 0.460, LinUCB 0.428, PPO 0.454, random 0.136 (official v2 seed 0: 0.656 / 0.460 /
0.428 / 0.514 / 0.136; PPO differs by run-to-run torch nondeterminism).

Suggested live script: Startup chatbot (heuristic overspends, RL fits the budget) ->
Code assistant (only Lagrangian PPO meets all three) -> Head-to-head with 100 customers.
Quality / latency / cost come from the uncalibrated simulator, not real hardware.

## Simulator toggle
The header switch flips between two simulators (`webdemo/sim_versions.py`):
- **Placeholder numbers**: the original tables (`results_frozen_v2_test`).
- **Real data (calibrated)**: constants refit to 29 public references (PR #5,
  `docs/EXTERNAL_REFERENCE_CHECK.md`).

Each mode has its own agents, trained on that simulator (`webdemo/models/<sim>/`), so
agents are always scored by the simulator they learned on. Retrain with:

```bash
python -m webdemo.pretrain --sim placeholder
python -m webdemo.pretrain --sim calibrated
```

On the full test set the saved agents reproduce official seed 0 meets-all in both modes
(placeholder: Lagrangian PPO 0.657, heuristic 0.460, LinUCB 0.428, random 0.136, PPO 0.454;
calibrated: 0.071 / 0.049 / 0.159 / 0.018, PPO 0.058, all identical to the recalibrated run).

The head-to-head screen adds a **Best possible** row: the share of customers for whom any
1-2 tier setup meets all three requirements (about 86% placeholder, about 35% calibrated).

Suggested live script: Code assistant in placeholder mode (only Lagrangian PPO meets all
three, 70B on H100 in 0.9 s) -> flip the toggle (the same setups now take 12-24 s, so no
setup can hit a 3 s limit) -> head-to-head with 100 customers in both modes.
