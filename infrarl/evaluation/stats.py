"""Statistical comparison of agents on paired, per-profile results.

Friedman rank test for a global difference across methods, then pairwise Wilcoxon
signed-rank tests (Holm-corrected) with mean difference as effect size, alpha=0.05.
Expects a long dataframe with columns: agent, seed, profile_idx, <metric>.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def compare_agents(df: pd.DataFrame, metric: str = "reward") -> pd.DataFrame:
    rows = []
    for seed, d in df.groupby("seed"):
        wide = d.pivot(index="profile_idx", columns="agent", values=metric)
        agents = list(wide.columns)
        if len(agents) >= 3:
            chi2, p = stats.friedmanchisquare(*[wide[a].values for a in agents])
            rows.append(dict(seed=seed, metric=metric, test="friedman", a="*", b="*",
                             statistic=chi2, p_raw=p, p_holm=p, mean_diff=np.nan))
        pairs = list(itertools.combinations(agents, 2))
        raw, recs = [], []
        for a, b in pairs:
            diff = wide[a].values - wide[b].values
            if np.allclose(diff, 0):
                stat, p = 0.0, 1.0
            else:
                stat, p = stats.wilcoxon(diff, zero_method="wilcox")
            raw.append(p)
            recs.append((a, b, stat, p, float(diff.mean())))
        for (a, b, stat, p, md), ph in zip(recs, holm(raw) if raw else []):
            rows.append(dict(seed=seed, metric=metric, test="wilcoxon", a=a, b=b,
                             statistic=stat, p_raw=p, p_holm=ph, mean_diff=md))
    return pd.DataFrame(rows)
