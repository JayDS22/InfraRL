# InfraRL

Reinforcement Learning for Use-Case-Driven AI Infrastructure Recommendation.
MSAI/MSML 635, University of Maryland, Fall 2026.

![architecture](docs/architecture.png)

## Team

- Jay - RL algorithms
- Vishnu - RL engineering
- Anupama - simulator and evaluation

## Deadlines

- Interim Report: Oct 26
- Presentation: last Monday class (Dec 1 or Dec 8)
- Final Report: Dec 7

## Setup

```bash
pip install -r requirements.txt
```

See `proposal.pdf` and `ROADMAP.md` for details. Dataflow and RL loop diagrams in `docs/`.

## Run

```bash
pytest -q tests
python demo.py                                              # original smoke test
python -m infrarl.evaluation.run_experiment --seeds 3 --out results
streamlit run infrarl/app.py
```

See `docs/EVALUATION.md` for the protocol, metrics and how to plug in agents.

Status: simulator tables are uncalibrated placeholders (3-config residual check pending).
Random, heuristic, context-free and contextual (LinUCB) bandit baselines, the evaluation harness,
ablations and the Streamlit demo run end to end. Q-learning, DQN, PPO and the online-adaptation
loop are not implemented yet; see `docs/EVALUATION.md` for how to plug them in.
