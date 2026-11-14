# Calibration provenance and coverage

No independently verified, source-matched benchmark values have been supplied in
this checkout. Reference numbers are intentionally blank. `PENDING_SOURCE` is not
a measured reference. Original infeasible rows retain `ORIGINAL_INFEASIBLE`.

| Primary investigation | Quality | Latency | Monthly service cost |
|---|---|---|---|
| CPU / 3B / INT8 | Missing named checkpoint/task | Missing CPU/runtime/workload | Missing provider/region/replica assumptions |
| A10 24 GB / 13B / INT8 | Missing named checkpoint/task | Missing matched inference data | Missing matched price and workload |
| A100 80 GB / 70B / INT8 | Missing named checkpoint/task | Missing runtime/KV-cache fit | Missing matched price and workload |

CPU/3B/INT8 is also the original CPU candidate. Two original FP16 anchors are
weights-infeasible: A10/13B needs 26 GB >24 GB; A100/70B needs 140 GB >80 GB.
Revised INT8 candidates are weights-only feasible, not demonstrated serving fits.

Candidate primary sources from the supplied guide, **not independently retrieved
or accepted as matched records in this audit**:

- Provider prices: https://nebius.com/prices — confirm exact hardware, provider,
  region, GPU count and effective date; an hourly list price is not monthly cost.
- Inference results: https://github.com/mlcommons/inference_results_v5.1 — match
  exact model, precision, GPU/VRAM, runtime, batch/concurrency and token lengths.
  Aggregate batched throughput cannot validate mean request latency directly.
- Metric definitions: https://docs.vllm.ai/en/stable/benchmarking/cli/ — distinguish
  E2E mean/p95, TTFT, TPOT, and aggregate/single-stream throughput.
- Hardware inventory: https://hpcc.umd.edu/home/clusters/zaratan/ — the supplied
  guide identifies Zaratan A100 40 GiB versus simulated A100 80 GB. Not interchangeable.

No specific quality benchmark can be matched until size buckets are mapped to
named checkpoints, use-case task/dataset and exact metric. Arena normalization,
MMLU accuracy and HumanEval pass@1 are different measurements. Do not combine
unrelated quality/latency/price rows and call them a complete matched benchmark.

## Filling and reviewing records

Each required primary configuration declares quality, latency and cost: nine
required cells. Record source URL, retrieval/publication/effective dates, provider,
region, exact hardware/VRAM/count, checkpoint, quantizer, runtime, workload,
batch/concurrency, input/output tokens, statistic and units. Explicit simulator
checkpoint/hardware/metric mappings must be approved and agree with the reference;
never fill mappings merely to make the arithmetic run. Quality uses fraction,
latency uses ms, and cost uses USD/month under identical replica/workload assumptions.
Zero/missing references or incompatible definitions cannot pass a relative gate.

Separate calibration rows from independent `holdout` rows; do not tune and validate
on the same points. No fitting is implemented. Constants, rewards, profiles and
masks remain unchanged. A required-axis PASS requires <=15% absolute relative
error; complete coverage is separate from holdout/generalization evidence.

Runner outcomes map PASS→passed, FAIL→failed, INFEASIBLE→infeasible,
MISSING_REFERENCE→unverified, NOT_COMPARABLE→incomparable. CLI exit 0 means complete
required pass, 1 means confirmed failure/required infeasibility, 2 means incomplete.
Slurm records the reporting operation separately from this scientific gate.
