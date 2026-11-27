# InfraRL live demo

A one-page web demo that runs the **real trained agents** through the InfraRL simulator.
Nothing on screen is precomputed: each click runs the saved agents step by step and scores
their stack with the simulator.

```bash
# from the repo root, using the project .venv (needs starlette + uvicorn, already installed)
python -m webdemo.pretrain     # only if webdemo/models/ is empty; ~1 min on a laptop CPU
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

## Which simulator the models match
The saved models in `webdemo/models/` were trained on the simulator constants on `main` as
of 2026-11-26 (the original placeholder tables). If the recalibration PR (real-data constants)
is merged, retrain them so the agents match the simulator that scores them:

```bash
python -m webdemo.pretrain    # ~1 minute; overwrites webdemo/models/*.pkl
```

The presets' "story" (which agent wins) depends on the simulator; re-check them after retraining.
