"""End-to-end experiment driver: train on synthetic profiles, evaluate greedily on a
held-out set, then write episodes / summaries / curves / stats / figures / ablations.

    python -m infrarl.evaluation.run_experiment --seeds 3 --out results
    python -m infrarl.evaluation.run_experiment --agents dqn,heuristic,random --ablate dqn
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from ..agents.registry import REGISTRY
from ..simulator.environment import InfraEnv
from ..simulator.profiles import generate_profiles
from . import plots
from .ablations import ABLATIONS, AblatedInfraEnv
from .experiments import episode_rows, evaluate, summarize, train, write_episodes
from .stats import compare_agents

TEST_SEED = 1000
METRICS = ["avg_quality", "quality_attainment", "quality_compliance", "budget_compliance",
           "sla_compliance", "meets_all", "mean_reward", "mean_cost_usd"]


def run_one(spec, seed, train_profiles, test_profiles, env_factory, epochs=None, arm_filter=None):
    env = env_factory(train_profiles, seed)
    agent = spec.factory(seed)
    if arm_filter is not None and hasattr(agent, "set_arm_filter"):
        agent.set_arm_filter(arm_filter)
    n_epochs = spec.train_epochs if epochs is None else (epochs if spec.train_epochs else 0)
    curve = train(env, agent, train_profiles, n_epochs, seed, eval_profiles=test_profiles) if n_epochs else []
    return agent, curve, evaluate(env, agent, test_profiles)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", default=",".join(REGISTRY))
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=None, help="override train epochs for learning agents")
    ap.add_argument("--ablate", default="bandit_contextual", help="agent to ablate ('' to skip)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(f"{args.out}/figures", exist_ok=True)
    names = [a for a in args.agents.split(",") if a]
    unknown = [a for a in names if a not in REGISTRY]
    if unknown:
        raise SystemExit(f"unknown agents {unknown}; registered: {list(REGISTRY)}")
    test_profiles = generate_profiles(args.n_test, seed=TEST_SEED)
    plain_env = lambda profs, seed: InfraEnv(profs, seed=seed)

    ep_rows, summ_rows, curve_rows = [], [], []
    for name in names:
        spec = REGISTRY[name]
        for seed in range(args.seeds):
            train_profiles = generate_profiles(args.n_train, seed=2000 + seed)
            _, curve, results = run_one(spec, seed, train_profiles, test_profiles, plain_env, args.epochs)
            ep_rows += list(episode_rows(name, seed, results))
            summ_rows.append({"agent": name, "seed": seed, **summarize(results)})
            curve_rows += [{"agent": name, "seed": seed, **c} for c in curve]
            print(f"[{name} seed {seed}] reward={summ_rows[-1]['mean_reward']:.3f} "
                  f"budget={summ_rows[-1]['budget_compliance']:.3f} sla={summ_rows[-1]['sla_compliance']:.3f}", flush=True)

    write_episodes(f"{args.out}/episodes.csv", ep_rows)
    episodes = pd.read_csv(f"{args.out}/episodes.csv")
    summary = pd.DataFrame(summ_rows)
    summary.to_csv(f"{args.out}/summary_per_seed.csv", index=False)
    agg = summary.groupby("agent")[METRICS].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg.to_csv(f"{args.out}/summary.csv")
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(f"{args.out}/curves.csv", index=False)

    stat_df = pd.concat([compare_agents(episodes, "reward"), compare_agents(episodes, "cost_usd")], ignore_index=True)
    stat_df.to_csv(f"{args.out}/stats.csv", index=False)

    plots.plot_pareto(summary, f"{args.out}/figures/pareto_quality_vs_cost.png")
    plots.plot_heatmap(episodes, f"{args.out}/figures/heatmap_meets_all_by_use_case.png")
    plots.plot_compliance(summary, f"{args.out}/figures/compliance.png")
    plots.plot_curves(curves, f"{args.out}/figures/learning_curves.png")

    if args.ablate:
        spec = REGISTRY[args.ablate]
        rows = []
        for variant in ["full", *ABLATIONS]:
            ab = ABLATIONS.get(variant)
            factory = plain_env if ab is None else (lambda p, s, ab=ab: AblatedInfraEnv(ab, p, seed=s))
            for seed in range(args.seeds):
                train_profiles = generate_profiles(args.n_train, seed=2000 + seed)
                _, _, res = run_one(spec, seed, train_profiles, test_profiles, factory, args.epochs,
                                    arm_filter=ab.arm_ok if ab else None)
                rows.append({"agent": args.ablate, "variant": variant, "seed": seed, **summarize(res)})
        abl = pd.DataFrame(rows)
        abl.to_csv(f"{args.out}/ablations.csv", index=False)
        plots.plot_ablation(abl, f"{args.out}/figures/ablations.png")
        print("\nAblations (mean over seeds):")
        print(abl.groupby("variant")[["mean_cost_usd", "avg_quality", "quality_compliance", "sla_compliance",
                                      "budget_compliance", "mean_reward"]].mean().round(3).to_string())

    pd.set_option("display.width", 200)
    print("\nSummary (mean over seeds):")
    print(summary.groupby("agent")[METRICS].mean().round(3).to_string())
    fr = stat_df[(stat_df.test == "friedman") & (stat_df.metric == "reward")]
    if len(fr):
        print("\nFriedman (reward) p-values per seed:", fr.p_raw.apply(lambda p: f"{p:.2e}").tolist())


if __name__ == "__main__":
    main()
