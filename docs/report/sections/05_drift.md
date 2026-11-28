## 5. Drift adaptation

### 5.1 Setup

The frozen-battery evaluation in Section 4 holds the profile distribution
fixed. Real deployments do not. Customer workloads shift as new features
ship, budgets rebalance, and traffic composition evolves. We simulate a
piecewise-stationary drift regime (Garivier and Moulines 2011) over a
four-week horizon and compare a frozen policy against an online-learning
variant. Code: `infrarl/online/drift.py`.

Both agents are pretrained for 20 epochs on 500 stationary profiles using
a contextual LinUCB bandit, then cloned. The **static** copy runs frozen
on the drift stream; the **online** copy keeps updating its per-arm
sufficient statistics as new profiles arrive. Both see the same stream of
250 profiles per week for four weeks, three independent seeds.

The drift schedule is linear across three independent axes, interpolated
from the pre-drift baseline (week 0) to a target distribution at week 3:

- **Use-case mix.** Baseline uniform over six use cases shifts toward
  chatbot plus RAG (70 percent of traffic jointly by week 3), mimicking
  a product-launch spike.
- **Accuracy requirement.** Uniform(0.60, 0.90) shifts to
  Uniform(0.75, 0.95), a harder quality floor.
- **Monthly budget.** log-Uniform($1k, $30k) shifts to
  log-Uniform($500, $15k), a FinOps-driven cost tightening.

The static agent cannot observe the shift; it serves recommendations from
the pre-drift policy. The online agent re-estimates per-arm value under
the moving distribution.

### 5.2 Results

Table 4 reports mean per-profile reward by week, averaged over the three
seeds. Source: `results_frozen_v1/drift/drift.csv`.

**Table 4. Drift adaptation: mean weekly reward (n=3 seeds).**

| Week | Static (frozen) | Online (updating) | Online - Static |
|---|---|---|---|
| 0 (pre-drift) | 0.909 ± 0.154 | 0.912 ± 0.149 | +0.003 |
| 1 | 0.858 ± 0.170 | 0.880 ± 0.143 | +0.022 |
| 2 | 0.787 ± 0.232 | 0.849 ± 0.180 | +0.062 |
| 3 | 0.793 ± 0.230 | 0.861 ± 0.173 | **+0.068** |

Figure 5 (`results_frozen_v1/drift/drift.png`) plots the per-step reward
trajectory with the per-week means overlaid.

The static policy loses 0.116 of reward (12.8 percent relative) between
week 0 and week 3, with the largest drop between weeks 1 and 2 as the
use-case mix and accuracy requirement cross the midpoint of their
interpolation. The online policy loses only 0.051 (5.6 percent). By the
end of the horizon, the online agent carries a +0.068 reward advantage
and a markedly lower standard deviation across seeds, indicating its
re-estimated arm statistics have converged to the drifted distribution
within the available data budget.

### 5.3 Interpretation

The drift experiment establishes two points. First, a policy trained
on a stationary distribution degrades predictably when the deployment
distribution shifts; a 13 percent reward loss over four weeks is
substantial at production scale and would not be visible to a monitoring
dashboard tracking aggregate throughput. Second, the recovery mechanism
required is modest. The online LinUCB variant used here is not
hyperparameter-tuned to the drift schedule; a discounted LinUCB
(D-LinUCB, Russac, Vernade, Cappé 2019 [6]) with a tuned forgetting
factor would likely recover faster, as would the online-selector agent
registered under `agent_selector` in the agent registry. These are
flagged in Section 9 as the natural production-adaptation path: pair a
Section 4 ranking recommendation with an online layer that handles the
drift the batch evaluation cannot.

### 5.4 Caveats

The experiment pretrains only the bandit family, not PPO or Lagrangian
PPO. Online PPO updating requires rollout-buffer reuse policies that are
non-trivial in the piecewise-stationary regime and were not scoped. The
drift reward-weight schedule
(`_SHIFTED_REWARD_WEIGHTS = (0.3, 0.5, 0.2)` in
`infrarl/online/drift.py`) is one plausible FinOps reprioritization path;
other re-weighting paths would produce different magnitude gaps.
