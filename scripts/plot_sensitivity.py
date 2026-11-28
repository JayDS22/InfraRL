"""Figure: meets-all under +/-20% simulator perturbation (results_sensitivity_v1).

    python scripts/plot_sensitivity.py [--src results_sensitivity_v1/sensitivity.csv] [--out ...png]

Three panels (quality, latency, cost multiplier), one line per agent, mean over seeds
with a min-max band across seeds. Nominal (x = 1.0) is the frozen v2 setting.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
AGENTS = [  # (key, label, color); fixed order, color follows the agent
    ("lagrangian_ppo", "Lagrangian PPO", "#2a78d6"),
    ("ppo", "PPO", "#eb6834"),
    ("heuristic", "Heuristic", "#1baf7a"),
    ("bandit_contextual", "LinUCB", "#a3a29c"),  # context series in neutral gray
]
AXES = [("quality", "q", "Quality scores"), ("latency", "l", "Latency"), ("cost", "c", "Cost")]
FACTORS = [0.8, 0.9, 1.0, 1.1, 1.2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results_sensitivity_v1/sensitivity.csv")
    ap.add_argument("--out", default="results_sensitivity_v1/sensitivity_meets_all.png")
    args = ap.parse_args()
    df = pd.read_csv(args.src)

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, axs = plt.subplots(1, 3, figsize=(13, 5.0), sharey=True, gridspec_kw={"width_ratios": [1, 1, 1.3]}, facecolor=SURFACE)
    for ax, (axis, prefix, title) in zip(axs, AXES):
        ax.set_facecolor(SURFACE)
        for key, label, color in AGENTS:
            pts = []
            for f in FACTORS:
                tag = "nominal" if f == 1.0 else f"{prefix}{f}"
                v = df[(df.agent == key) & (df.perturbation == tag)].meets_all
                pts.append((f, v.mean(), v.min(), v.max()))
            p = pd.DataFrame(pts, columns=["f", "mean", "lo", "hi"])
            z = 3 if key == "lagrangian_ppo" else 2
            ax.fill_between(p.f, p.lo, p.hi, color=color, alpha=0.12, linewidth=0, zorder=z)
            ax.plot(p.f, p["mean"], color=color, lw=2, zorder=z + 1)
            ax.scatter(p.f, p["mean"], s=36, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=z + 2)
            if axis == "cost":  # direct labels where the lines are well separated
                ax.annotate(label, (1.2, p["mean"].iloc[-1]), xytext=(10, {"ppo": 5, "heuristic": -5}.get(key, 0)), textcoords="offset points",
                            va="center", fontsize=10, color=INK)
        ax.axvline(1.0, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
        ax.axhline(0.75, color=INK2, lw=1, ls=(0, (1, 2)), zorder=1)
        ax.set_title(f"{title} ×0.8 to ×1.2", color=INK, fontsize=12, loc="left")
        ax.set_xticks(FACTORS, ["−20%", "−10%", "as built", "+10%", "+20%"])
        ax.set_ylim(0, 0.85)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    # label the 0.75 target in the cost panel, where no line passes near it
    axs[1].text(0.8, 0.765, "target 0.75", fontsize=9, color=INK2)
    axs[0].set_ylabel("Meets all 3 constraints (share of customers)")
    for ax in axs:
        ax.set_xlim(0.78, 1.22)
    axs[2].set_xlim(0.78, 1.38)
    handles = [plt.Line2D([], [], color=c, lw=2, marker="o", ms=6) for _, _, c in AGENTS]
    fig.legend(handles, [l for _, l, _ in AGENTS], loc="upper left", ncol=4, frameon=False,
               bbox_to_anchor=(0.005, 0.935), labelcolor=INK, fontsize=10.5, handlelength=1.8)
    fig.suptitle("Ranking holds under cost and latency error; it depends on the quality numbers",
                 x=0.01, ha="left", fontsize=14, color=INK, y=0.985)
    fig.text(0.01, 0.005, "Agents trained on the nominal simulator, then tested with one output scaled ×0.8 to ×1.2. "
             "Line = mean of 3 seeds, band = seed min–max. 1000 held-out test customers. Source: results_sensitivity_v1.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.045, 1, 0.885))
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(args.out)


if __name__ == "__main__":
    main()
