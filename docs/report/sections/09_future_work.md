## 9. Future work

- **Empirical calibration against vLLM on Zaratan A100 40 GB.** Rebaseline
  the simulator's hardware table from the simulated A100 80 GB SKU to the
  Zaratan A100 40 GB SKU and run matched workloads under vLLM with
  recorded TTFT, TPOT, and E2E latency statistics. A single matched run
  converts at least one row of the residual gate from `NOT_COMPARABLE`
  to a numeric residual and tightens the Section 7 quality-axis bound.
- **Online calibration from live telemetry.** The D-LinUCB agent
  (Section 3.4) already handles piecewise-stationary reward drift via its
  discount factor; wiring it to a real reward signal from a production
  gateway is a tractable next step for a pilot. The drift harness in
  `infrarl/online/drift.py` is the integration seam.
- **Multi-tenant extension.** Competing profiles share a finite GPU pool;
  the policy schedules allocations across tenants rather than selecting
  independently per profile. This moves the formulation from single-profile
  stack selection into true multi-agent resource contention with
  cross-tenant SLA coupling.
