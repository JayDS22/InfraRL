# Calibration residual assessment

Gate: **UNVERIFIED**; complete required coverage: False.
Tolerance: <=15% only for independently matched comparable axes. Original infeasible tests are excluded from required coverage.

| Configuration | Axis | Prediction | Reference | Units | Status | Reason |
|---|---|---|---|---|---|---|
| cpu_3b_int8 | quality |  |  | fraction | MISSING_REFERENCE | No independently verified reference |
| cpu_3b_int8 | latency |  |  | ms | MISSING_REFERENCE | No independently verified reference |
| cpu_3b_int8 | cost |  |  | USD/month | MISSING_REFERENCE | No independently verified reference |
| a10_13b_int8 | quality |  |  | fraction | MISSING_REFERENCE | No independently verified reference |
| a10_13b_int8 | latency |  |  | ms | MISSING_REFERENCE | No independently verified reference |
| a10_13b_int8 | cost |  |  | USD/month | MISSING_REFERENCE | No independently verified reference |
| a100_70b_int8 | quality |  |  | fraction | MISSING_REFERENCE | No independently verified reference |
| a100_70b_int8 | latency |  |  | ms | MISSING_REFERENCE | No independently verified reference |
| a100_70b_int8 | cost |  |  | USD/month | MISSING_REFERENCE | No independently verified reference |
| a10_13b_fp16 | quality |  |  | fraction | INFEASIBLE | Weights-only memory exceeds one device; never estimated or fitted |
| a10_13b_fp16 | latency |  |  | ms | INFEASIBLE | Weights-only memory exceeds one device; never estimated or fitted |
| a10_13b_fp16 | cost |  |  | USD/month | INFEASIBLE | Weights-only memory exceeds one device; never estimated or fitted |
| a100_70b_fp16 | quality |  |  | fraction | INFEASIBLE | Weights-only memory exceeds one device; never estimated or fitted |
| a100_70b_fp16 | latency |  |  | ms | INFEASIBLE | Weights-only memory exceeds one device; never estimated or fitted |
| a100_70b_fp16 | cost |  |  | USD/month | INFEASIBLE | Weights-only memory exceeds one device; never estimated or fitted |

Unmatched required axes: cpu_3b_int8_quality, cpu_3b_int8_latency, cpu_3b_int8_cost, a10_13b_int8_quality, a10_13b_int8_latency, a10_13b_int8_cost, a100_70b_int8_quality, a100_70b_int8_latency, a100_70b_int8_cost

Independent holdout cells: 0. A complete anchor pass alone does not establish generalization.
