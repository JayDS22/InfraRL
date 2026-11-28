"""Re-render the deck's hero figures with the editorial-light palette.

Reads existing CSVs in `results_frozen_v2_test/` and `results_sensitivity_v1/`
and writes re-styled PNGs alongside the originals as `*_editorial.png`.
Keeps the originals intact (the report body references them). The deck
points at the `_editorial.png` versions via theme-matched paths.

Run: `uv run --with numpy --with pandas --with matplotlib python scripts/render_editorial_figures.py`
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CREAM = "#F5F0E8"
INK = "#1C1917"
ACCENT = "#7C2D12"
MUTED = "#78716C"
SOFT = "#D6D3D1"

AGENT_COLORS = {
    "lagrangian_ppo": ACCENT,
    "ppo": "#B45309",
    "heuristic": "#44403C",
    "bandit_contextual": "#78716C",
    "bandit_context_free": "#A8A29E",
    "bandit_discounted": "#A8A29E",
    "q_learning": "#D6D3D1",
    "random": "#D6D3D1",
}

RC = {
    "figure.facecolor": CREAM,
    "axes.facecolor": CREAM,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "font.family": ["Inter", "Helvetica Neue", "Arial", "sans-serif"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "savefig.facecolor": CREAM,
    "savefig.edgecolor": CREAM,
}


def _color(agent: str) -> str:
    return AGENT_COLORS.get(agent, MUTED)


def pareto_frontier(points):
    keep = []
    for i, (x, y) in enumerate(points):
        if not any(
            (x2 <= x and y2 >= y) and (x2 < x or y2 > y)
            for j, (x2, y2) in enumerate(points) if j != i
        ):
            keep.append(i)
    return keep


def render_pareto(summary_path: Path, out: Path) -> None:
    summary = pd.read_csv(summary_path)
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for _, row in summary.iterrows():
            agent = row["agent"]
            ax.errorbar(
                row["mean_cost_usd_mean"], row["avg_quality_mean"],
                xerr=row["mean_cost_usd_std"], yerr=row["avg_quality_std"],
                fmt="o", markersize=10, color=_color(agent),
                ecolor=_color(agent), elinewidth=1.2, capsize=0,
                label=agent, alpha=0.95,
            )
            ax.annotate(
                agent, (row["mean_cost_usd_mean"], row["avg_quality_mean"]),
                xytext=(8, 4), textcoords="offset points",
                fontsize=9, color=INK,
            )
        pts = list(zip(summary["mean_cost_usd_mean"], summary["avg_quality_mean"]))
        front = sorted(pareto_frontier(pts), key=lambda i: pts[i][0])
        if len(front) > 1:
            ax.plot(
                [pts[i][0] for i in front], [pts[i][1] for i in front],
                "--", color=INK, lw=0.8, alpha=0.4,
            )
        ax.set_xlabel("MEAN MONTHLY COST (USD)", labelpad=12, fontsize=10)
        ax.set_ylabel("MEAN RECOMMENDED-STACK QUALITY", labelpad=12, fontsize=10)
        fig.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)


def render_learning_curves(curves_path: Path, out: Path) -> None:
    curves = pd.read_csv(curves_path)
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(8, 5))
        for agent, sub in curves.groupby("agent"):
            if "eval_reward" not in sub.columns or sub["eval_reward"].isna().all():
                continue
            mean = sub.groupby("epoch")["eval_reward"].mean()
            ax.plot(mean.index, mean.values, color=_color(agent),
                    lw=2 if agent in ("lagrangian_ppo", "ppo") else 1.2,
                    alpha=0.95, label=agent)
        ax.set_xlabel("EPOCH", labelpad=12, fontsize=10)
        ax.set_ylabel("MEAN EVAL REWARD", labelpad=12, fontsize=10)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)


def render_sensitivity(sens_path: Path, out: Path) -> None:
    sens = pd.read_csv(sens_path)
    with plt.rc_context(RC):
        axes_order = ("quality", "latency", "cost")
        fig, axes = plt.subplots(1, len(axes_order), figsize=(13, 4.5), sharey=True)
        nominal = sens[sens["perturbation"] == "nominal"].groupby("agent")["meets_all"].mean()
        for ax, axis in zip(axes, axes_order):
            mult_col = f"{axis}_mult"
            others = [f"{a}_mult" for a in axes_order if a != axis]
            sub = sens[(sens["perturbation"] == "nominal") |
                       ((sens[mult_col] != 1.0) & (sens[others[0]] == 1.0) & (sens[others[1]] == 1.0))]
            pivot = sub.groupby(["agent", mult_col])["meets_all"].mean().unstack(mult_col)
            if 1.0 not in pivot.columns:
                pivot[1.0] = pivot.index.map(nominal)
            pivot = pivot.reindex(sorted(pivot.columns), axis=1)
            for agent, row in pivot.iterrows():
                ax.plot(row.index, row.values, marker="o", markersize=7,
                        color=_color(agent), lw=1.8 if agent == "lagrangian_ppo" else 1.2,
                        alpha=0.95, label=agent)
            ax.axvline(1.0, color=INK, lw=0.5, linestyle=":", alpha=0.5)
            ax.set_xlabel(f"{axis.upper()} MULTIPLIER", labelpad=10, fontsize=10)
            ax.set_title(f"perturb {axis}", fontsize=12, pad=12)
        axes[0].set_ylabel("MEETS ALL", labelpad=12, fontsize=10)
        axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    render_pareto(
        root / "results_frozen_v2_test" / "summary.csv",
        root / "results_frozen_v2_test" / "figures" / "pareto_quality_vs_cost_editorial.png",
    )
    render_learning_curves(
        root / "results_frozen_v2_test" / "curves.csv",
        root / "results_frozen_v2_test" / "figures" / "learning_curves_editorial.png",
    )
    render_sensitivity(
        root / "results_sensitivity_v1" / "sensitivity.csv",
        root / "results_sensitivity_v1" / "sensitivity_editorial.png",
    )
    print("rendered 3 editorial-palette figures")


if __name__ == "__main__":
    main()
