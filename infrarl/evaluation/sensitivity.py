"""Simulator-perturbation robustness analysis.

Each agent is trained once per seed under the nominal estimator, then evaluated
on the held-out test profiles under multiplicative perturbations of
estimate()'s outputs. Produces one CSV + JSON per run, consumed by
plots.plot_sensitivity() for the ranking-stability figure.

Perturbation is applied by rebinding the imported `estimate` in
infrarl.simulator.environment; the `from .estimator import ... estimate`
there creates a module-local binding that patching estimator.estimate alone
would not affect.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from ..agents.registry import REGISTRY
from ..simulator import environment as env_module
from ..simulator.environment import InfraEnv
from ..simulator.profiles import generate_profiles
from .experiments import evaluate, summarize, train
from .run_experiment import TEST_SEED


@dataclass(frozen=True)
class Perturbation:
    quality_mult: float = 1.0
    latency_mult: float = 1.0
    cost_mult: float = 1.0

    @property
    def is_nominal(self) -> bool:
        return self.quality_mult == 1.0 and self.latency_mult == 1.0 and self.cost_mult == 1.0

    @property
    def label(self) -> str:
        if self.is_nominal:
            return "nominal"
        parts = []
        if self.quality_mult != 1.0:
            parts.append(f"q{self.quality_mult:g}")
        if self.latency_mult != 1.0:
            parts.append(f"l{self.latency_mult:g}")
        if self.cost_mult != 1.0:
            parts.append(f"c{self.cost_mult:g}")
        return "_".join(parts)


@contextlib.contextmanager
def perturbed_estimate(perturbation: Perturbation) -> Iterator[None]:
    original = env_module.estimate

    def patched(stack, profile):
        quality, latency, cost = original(stack, profile)
        return (
            max(0.0, min(1.0, quality * perturbation.quality_mult)),
            latency * perturbation.latency_mult,
            cost * perturbation.cost_mult,
        )

    env_module.estimate = patched
    try:
        yield
    finally:
        env_module.estimate = original


def perturbation_grid(axes: tuple[str, ...], factors: tuple[float, ...]) -> list[Perturbation]:
    valid = {"quality", "latency", "cost"}
    unknown = set(axes) - valid
    if unknown:
        raise ValueError(f"unknown axes: {sorted(unknown)}; valid: {sorted(valid)}")
    grid: list[Perturbation] = [Perturbation()]
    for axis in axes:
        for factor in factors:
            if factor == 1.0:
                continue
            grid.append(Perturbation(**{f"{axis}_mult": float(factor)}))
    return grid


def run_sensitivity(agents: tuple[str, ...], seeds: int, factors: tuple[float, ...],
                    axes: tuple[str, ...], n_train: int, n_test: int, out: Path) -> None:
    """Train each agent once per seed, then re-evaluate under each perturbation."""
    out.mkdir(parents=True, exist_ok=True)
    test_profiles = generate_profiles(n_test, seed=TEST_SEED)
    grid = perturbation_grid(axes, factors)

    rows: list[dict] = []
    for agent_name in agents:
        if agent_name not in REGISTRY:
            raise KeyError(f"unknown agent: {agent_name}")
        spec = REGISTRY[agent_name]
        for seed in range(seeds):
            train_profiles = generate_profiles(n_train, seed=2000 + seed)
            env = InfraEnv(train_profiles, seed=seed)
            agent = spec.factory(seed)
            if spec.train_epochs:
                train(env, agent, train_profiles, spec.train_epochs, seed)
            for p in grid:
                with perturbed_estimate(p):
                    metrics = summarize(evaluate(env, agent, test_profiles))
                rows.append({
                    "agent": agent_name, "seed": seed, "perturbation": p.label,
                    "quality_mult": p.quality_mult,
                    "latency_mult": p.latency_mult,
                    "cost_mult": p.cost_mult,
                    **metrics,
                })
            print(f"[sensitivity] {agent_name} seed={seed} points={len(grid)}", flush=True)

    with (out / "sensitivity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "sensitivity.json").write_text(json.dumps({
        "agents": list(agents),
        "seeds": seeds,
        "axes": list(axes),
        "factors": list(factors),
        "n_train": n_train,
        "n_test": n_test,
        "test_seed": TEST_SEED,
        "perturbations": [asdict(p) for p in grid],
    }, indent=2, sort_keys=True) + "\n")


def main() -> None:
    from infrarl import require_compute_allocation
    require_compute_allocation()
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", required=True,
                    help="comma-separated inventory, e.g. heuristic,ppo,lagrangian_ppo")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--factors", default="0.8,0.9,1.1,1.2",
                    help="multiplicative factors; 1.0 is always included as the nominal baseline")
    ap.add_argument("--axes", default="quality,latency,cost",
                    help="axes to perturb, any subset of quality,latency,cost")
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--out", default="results_sensitivity")
    args = ap.parse_args()
    run_sensitivity(
        agents=tuple(args.agents.split(",")),
        seeds=args.seeds,
        factors=tuple(float(f) for f in args.factors.split(",")),
        axes=tuple(args.axes.split(",")),
        n_train=args.n_train,
        n_test=args.n_test,
        out=Path(args.out),
    )


if __name__ == "__main__":
    main()
