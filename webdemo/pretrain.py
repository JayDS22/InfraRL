"""Train the demo agents once and save them, so the demo answers instantly.

    python -m webdemo.pretrain            # all demo agents, seed 0 (~10 min on a laptop CPU)
    python -m webdemo.pretrain --agents heuristic,random

Same training recipe as the evaluation harness: 1000 synthetic training profiles
(seed 2000 + seed), the registry's epoch budget per agent. Models land in webdemo/models/.
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

from infrarl.agents.registry import REGISTRY
from infrarl.evaluation.experiments import train
from infrarl.simulator.environment import InfraEnv
from infrarl.simulator.profiles import generate_profiles

DEMO_AGENTS = ["lagrangian_ppo", "ppo", "heuristic", "bandit_contextual", "random"]
MODEL_DIR = Path(__file__).resolve().parent / "models"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", default=",".join(DEMO_AGENTS))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    MODEL_DIR.mkdir(exist_ok=True)
    profiles = generate_profiles(1000, seed=2000 + args.seed)
    for name in args.agents.split(","):
        spec = REGISTRY[name]
        agent = spec.factory(args.seed)
        t0 = time.time()
        if spec.train_epochs:
            train(InfraEnv(profiles, seed=args.seed), agent, profiles, spec.train_epochs, seed=args.seed)
        with open(MODEL_DIR / f"{name}.pkl", "wb") as f:
            pickle.dump(agent, f)
        print(f"[{name}] epochs={spec.train_epochs} took {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
