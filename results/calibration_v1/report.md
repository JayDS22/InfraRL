# Calibration residual assessment

Gate: **FAILED**; complete required coverage: False.
Tolerance: <=15% only for independently matched comparable axes. Original infeasible tests are excluded from required coverage.

| Configuration | Axis | Prediction | Reference | Units | Status | Reason |
|---|---|---|---|---|---|---|
| cpu_3b_int8 | quality |  | 0.634 | fraction | NOT_COMPARABLE | Incomplete provenance, checkpoint mapping or incompatible metric/configuration |
| cpu_3b_int8 | latency | 1442.3076923076924 | 8000 | ms | FAIL | Matched-axis relative error; tolerance <=15% |
| cpu_3b_int8 | cost | 288.0 | 2056.32 | USD/month | FAIL | Matched-axis relative error; tolerance <=15% |
| a10_13b_int8 | quality |  | 0.546 | fraction | NOT_COMPARABLE | Incomplete provenance, checkpoint mapping or incompatible metric/configuration |
| a10_13b_int8 | latency | 520.8333333333334 | 2586 | ms | FAIL | Matched-axis relative error; tolerance <=15% |
| a10_13b_int8 | cost | 720.0 | 724.32 | USD/month | PASS | Matched-axis relative error; tolerance <=15% |
| a100_70b_int8 | quality |  | 0.836 | fraction | NOT_COMPARABLE | Incomplete provenance, checkpoint mapping or incompatible metric/configuration |
| a100_70b_int8 | latency | 1009.6153846153845 | 4286 | ms | FAIL | Matched-axis relative error; tolerance <=15% |
| a100_70b_int8 | cost | 1728.0 | 2470.14 | USD/month | FAIL | Matched-axis relative error; tolerance <=15% |
| a10_13b_fp16 | quality |  |  | fraction | INFEASIBLE | Weights-only memory exceeds one device; never estimated or fitted |
| a10_13b_fp16 | latency |  |  | ms | INFEASIBLE | Weights-only memory exceeds one device; never estimated or fitted |
| a10_13b_fp16 | cost |  |  | USD/month | INFEASIBLE | Weights-only memory exceeds one device; never estimated or fitted |
| a100_70b_fp16 | quality |  |  | fraction | INFEASIBLE | Weights-only memory exceeds one device; never estimated or fitted |
| a100_70b_fp16 | latency |  |  | ms | INFEASIBLE | Weights-only memory exceeds one device; never estimated or fitted |
| a100_70b_fp16 | cost |  |  | USD/month | INFEASIBLE | Weights-only memory exceeds one device; never estimated or fitted |

Unmatched required axes: cpu_3b_int8_quality, cpu_3b_int8_latency, cpu_3b_int8_cost, a10_13b_int8_quality, a10_13b_int8_latency, a100_70b_int8_quality, a100_70b_int8_latency, a100_70b_int8_cost

Independent holdout cells: 0. A complete anchor pass alone does not establish generalization.
