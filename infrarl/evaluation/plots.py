"""Figures: Pareto (quality vs cost), per-use-case heatmap, learning curves,
compliance bars, ablation bars. All take tidy dataframes produced by run_experiment."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def pareto_frontier(points: list[tuple[float, float]]) -> list[int]:
    """Indices of non-dominated points (lower cost x, higher quality y)."""
    keep = []
    for i, (x, y) in enumerate(points):
        if not any((x2 <= x and y2 >= y) and (x2 < x or y2 > y) for j, (x2, y2) in enumerate(points) if j != i):
            keep.append(i)
    return keep


def plot_pareto(summary: pd.DataFrame, path: str, quality_col: str = "avg_quality") -> None:
    g = summary.groupby("agent")[[quality_col, "mean_cost_usd"]].agg(["mean", "std"]).fillna(0)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    pts = [(g.loc[a, ("mean_cost_usd", "mean")], g.loc[a, (quality_col, "mean")]) for a in g.index]
    for a, (x, y) in zip(g.index, pts):
        ax.errorbar(x, y, xerr=g.loc[a, ("mean_cost_usd", "std")], yerr=g.loc[a, (quality_col, "std")],
                    fmt="o", capsize=3, label=a)
    front = sorted(pareto_frontier(pts), key=lambda i: pts[i][0])
    if len(front) > 1:
        ax.plot([pts[i][0] for i in front], [pts[i][1] for i in front], "--", color="gray", lw=1, label="Pareto front")
    ax.set_xlabel("mean monthly cost (USD)")
    ax.set_ylabel("mean recommended-stack quality" if quality_col == "avg_quality" else quality_col)
    ax.set_title("Quality vs cost (mean ± std over seeds)")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_heatmap(episodes: pd.DataFrame, path: str, metric: str = "meets_all") -> None:
    d = episodes.copy()
    if metric == "meets_all":
        d["meets_all"] = d.within_budget & d.meets_sla.astype(bool) & d.meets_quality.astype(bool)
    pivot = d.groupby(["use_case", "agent"])[metric].mean().unstack("agent")
    fig, ax = plt.subplots(figsize=(1.3 * len(pivot.columns) + 2, 4))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, ax=ax, cbar_kws={"label": metric})
    ax.set_title(f"{metric} by use case")
    ax.set_xlabel("")
    ax.set_ylabel("")
    _save(fig, path)


def plot_curves(curves: pd.DataFrame, path: str) -> None:
    if curves.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for agent, d in curves.groupby("agent"):
        for col, ls in (("train_reward", "-"), ("eval_reward", "--")):
            if col not in d:
                continue
            g = d.groupby("epoch")[col].agg(["mean", "std"]).fillna(0)
            line, = ax.plot(g.index, g["mean"], ls, label=f"{agent} ({col.split('_')[0]})")
            ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], alpha=0.15, color=line.get_color())
    ax.set_xlabel("epoch (pass over training profiles)")
    ax.set_ylabel("mean terminal reward")
    ax.set_title("Learning curves (train = exploratory, eval = greedy held-out)")
    ax.legend(fontsize=7)
    _save(fig, path)


def plot_compliance(summary: pd.DataFrame, path: str) -> None:
    cols = ["budget_compliance", "sla_compliance", "quality_compliance", "meets_all"]
    g = summary.groupby("agent")[cols].agg(["mean", "std"]).fillna(0)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    w = 0.8 / len(cols)
    x = np.arange(len(g.index))
    for k, c in enumerate(cols):
        ax.bar(x + k * w, g[(c, "mean")], w, yerr=g[(c, "std")], capsize=2, label=c.replace("_", " "))
    ax.axhline(0.9, color="red", ls=":", lw=1)
    ax.text(len(g.index) - 0.5, 0.91, "90% target", color="red", fontsize=7, ha="right")
    ax.set_xticks(x + w * (len(cols) - 1) / 2, g.index, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction of held-out profiles")
    ax.set_title("Operational compliance (mean ± std over seeds)")
    ax.legend(fontsize=7, ncol=2)
    _save(fig, path)


def plot_ablation(abl: pd.DataFrame, path: str) -> None:
    metrics = ["mean_cost_usd", "quality_compliance", "sla_compliance", "budget_compliance"]
    g = abl.groupby("variant")[metrics].mean()
    base = g.loc["full"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 3.8))
    for ax, m in zip(axes, metrics):
        delta = (g[m] - base[m]).drop("full")
        ax.barh(delta.index, delta.values, color="tab:gray")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"Δ {m.replace('_', ' ')}", fontsize=9)
        ax.tick_params(labelsize=7)
    _save(fig, path)
