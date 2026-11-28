## 7. Simulator sensitivity

### 7.1 Method

The simulator is a structural lookup-table model, not a calibrated digital
twin. To defend the ranking in Section 4 against the "uncalibrated
simulator" critique, we perturb the simulator's `estimate()` output
multiplicatively on three independent axes (quality, latency, cost) and
re-evaluate each trained policy on the held-out test split (`test_seed =
1000`, `n_test = 1000`). Perturbation is applied by rebinding the imported
`estimate` symbol inside `infrarl.simulator.environment`, because
`from .estimator import ... estimate` in that module creates a module-local
binding that patching `estimator.estimate` alone would not affect
(`infrarl/evaluation/sensitivity.py:55-71`). Agents are trained once per
seed on the nominal estimator, then the perturbation is swapped in only for
test-time evaluation. This measures the sensitivity of the *learned
policy* to simulator miscalibration, not re-training sensitivity. Four
agents are evaluated (heuristic, bandit_contextual, ppo, lagrangian_ppo)
across three seeds and four non-identity multipliers per axis (0.8, 0.9,
1.1, 1.2) plus the nominal 1.0 baseline, perturbing one axis at a time.
Joint perturbation is deferred to Section 9. Figure 6 renders seed-mean
meets_all against multiplier for each axis.

### 7.2 Finding

- **Cost perturbation at plus-or-minus 20 percent: ranking fully
  preserved.** The ordering lagrangian_ppo > ppo > heuristic >
  bandit_contextual holds at every grid point. Lagrangian PPO's meets_all
  moves from 0.678 at `cost x 0.8` to 0.619 at `cost x 1.2`, a span of
  0.059. No lines cross.
- **Latency perturbation at plus-or-minus 20 percent: ranking fully
  preserved.** Same ordering across the grid. Lagrangian PPO moves from
  0.673 at `latency x 0.8` to 0.619 at `latency x 1.2`. PPO stays above
  the heuristic throughout.
- **Quality perturbation: ranking preserved only within plus-or-minus 10
  percent.** At `quality x 1.2`, PPO (0.795) overtakes Lagrangian PPO
  (0.724) on meets_all, because PPO's cheaper, lower-quality allocations
  begin to clear an inflated quality floor "for free" and the cost
  savings dominate the composite metric. The heuristic (0.705) and
  bandit_contextual (0.700) close in on Lagrangian PPO at this extreme.
  At `quality x 0.8`, Lagrangian PPO still leads (0.127) but the whole
  system collapses: PPO drops to 0.065, bandit_contextual to 0.023, and
  the heuristic to 0.000.

### 7.3 Interpretation

Under cost or latency miscalibration up to plus-or-minus 20 percent, the
Section 4 ranking is the operative recommendation without caveat. Under
quality miscalibration, the Lagrangian PPO advantage over vanilla PPO
depends on the simulator's quality numbers being correct within about
plus-or-minus 10 percent. Beyond that band the Lagrangian-versus-vanilla
choice becomes workload-dependent: a simulator that systematically
over-reports quality makes cheap stacks look compliant and rewards vanilla
PPO's cost-minimization; a simulator that under-reports quality still
favors the constrained policy. The implied calibration priority order is
quality first, latency second, cost third. This also motivates the
quality-axis focus of the three-anchor residual audit in Section 6.

### 7.4 Caveats

Perturbation is one axis at a time; joint perturbation (for example,
`quality x 0.8` and `cost x 1.2` together) is not swept and is deferred
to Section 9 future work. Only four agents are perturbed, not all eight.
The four excluded agents (random, bandit_context_free, bandit_discounted,
q_learning) sit well below the Lagrangian > PPO > heuristic cluster on
nominal meets_all; their inclusion would not change the rank-stability
story at the top of the leaderboard. Constrained-MDP framing and the
Lagrangian dual update follow Altman [1] and Stooke, Achiam, Abbeel [4];
the base PPO clip objective is Schulman et al. [2].
