## 6. Calibration residuals

### 6.1 Method

The calibration gate (`infrarl/simulator/calibration.py`) compares the
simulator's `estimate()` prediction against an independently-sourced public
reference for each of nine required cells: three configurations
(`cpu_3b_int8`, `a10_13b_int8`, `a100_70b_int8`) by three axes (quality,
latency, cost). A cell is comparable when its measurement type matches the
simulator's, its hardware VRAM matches `HARDWARE[hw]['memory_gb']` for GPU
rows, and full provenance is present (source URL, retrieval date,
publication date, provider, runtime, workload, statistic). Residuals
exceeding 15 percent relative error are reported as FAIL. The three
configurations were chosen to span the hardware-by-model matrix the
simulator exposes: a CPU small-model anchor, a mid-range A10 anchor, and
a high-end A100 70B anchor. All references are documented in
`data/benchmarks/calibration_references.csv`; the gate output lives in
`results/calibration_v1/`.

### 6.2 Results

**Table 3. Residual gate outcomes for nine required cells.**

| Configuration | Axis | Prediction | Reference | Units | Residual | Status |
|---|---|---|---|---|---|---|
| cpu_3b_int8 | quality | - | 0.634 | fraction | n/a | NOT_COMPARABLE |
| cpu_3b_int8 | latency | 1442 | 8000 | ms | 82.0% | FAIL |
| cpu_3b_int8 | cost | 288 | 2056 | USD/mo | 86.0% | FAIL |
| a10_13b_int8 | quality | - | 0.546 | fraction | n/a | NOT_COMPARABLE |
| a10_13b_int8 | latency | 521 | 2586 | ms | 79.9% | FAIL |
| a10_13b_int8 | cost | 720 | 724 | USD/mo | 0.6% | PASS |
| a100_70b_int8 | quality | - | 0.836 | fraction | n/a | NOT_COMPARABLE |
| a100_70b_int8 | latency | 1010 | 4286 | ms | 76.4% | FAIL |
| a100_70b_int8 | cost | 1728 | 2470 | USD/mo | 30.0% | FAIL |

Summary: one PASS (A10 cost), five FAIL (all three latency axes plus CPU
and A100 cost), three NOT_COMPARABLE (all quality axes). Gate outcome:
FAILED.

### 6.3 Findings

**Latency is systematically overestimated by the simulator by a factor of
roughly 4 to 5.5.** All three latency references land in the same
direction: the simulator predicts single-stream mean latency 4.4x
(cpu_3b_int8), 5.0x (a10_13b_int8), and 4.2x (a100_70b_int8) faster than
published vLLM and llama.cpp benchmarks for matched workloads. The
cross-tier consistency points to one root cause: the simulator's
`tps_13b_fp16` constants in `infrarl/simulator/hardware.py` (15 for CPU,
180 for A10, 500 for A100) reflect batched, continuous-batching throughput
rather than single-stream decode throughput. For a single request with no
concurrent load, real inference engines produce roughly 15 to 25 tokens
per second on CPU for a 3B INT8 model, 58 tokens per second on A10 for
13B INT8 via vLLM, and 35 tokens per second on A100 for 70B AWQ INT4 via
vLLM. The 4 to 5.5x gap is consistent with the ratio by which batched
throughput exceeds single-stream throughput in these engines.

**Cost is inconsistent across SKUs.** The A10 cell passes at 0.6 percent
residual, essentially spot-on for the published AWS g5.xlarge on-demand
rate. The CPU cell fails at 86 percent because the simulator's
`cost_per_hr_usd = 0.40` for CPU targets a smaller SKU (roughly
c7i.xlarge or m7i.large) than the SKU required to serve a 3B INT8 model
at the simulator's assumed throughput (c7i.16xlarge at $2.856/hr is the
closest match to the throughput assumption). The A100 cell fails at 30
percent because the simulator's `$2.40/hr` reflects a reserved or
multi-year committed rate; the public on-demand p4de.24xlarge rate
normalizes to $3.43/hr per A100-80GB GPU. Both CPU and A100 cost
residuals are a provisioning-regime mismatch rather than a modeling
error.

**Quality is structurally NOT_COMPARABLE.** The simulator's quality head
is a use-case specific fraction (chatbot, code generation, RAG, etc.)
derived from internal profile tables, not from any public benchmark.
Public LLM benchmarks (MMLU, HumanEval, HellaSwag, Open LLM Leaderboard
average) measure different constructs: knowledge recall, code synthesis,
commonsense inference. A direct residual against MMLU is a category
error. We cite the published values for each anchor
(Llama-3.2-3B-Instruct MMLU 0.634, Llama-2-13B MMLU 0.546,
Llama-3.1-70B-Instruct MMLU 0.836) for reader orientation, but the
calibration gate correctly marks them NOT_COMPARABLE.

### 6.4 Second-pass audit: 29 public references

The three-anchor audit above identified the direction of the error
(latency overestimated by ~5x) but could not rebaseline the simulator
because it had too few data points to refit the constants. A
follow-up audit against 29 public references (data provenance in
`data/benchmarks/external_references_20261127.csv`, methodology in
`docs/EXTERNAL_REFERENCE_CHECK.md`, re-runnable via
`python scripts/external_reference_check.py`) expanded the evidence
base and produced the refit values now shipped in
`infrarl/simulator/hardware.py` and `infrarl/simulator/models.py`.

The 29 references split as:
- 11 GPU/CPU hourly prices across AWS (via Vantage), Lambda, Nebius,
  Runpod. Prices were mostly already within the 15 percent gate: 6 of
  11 PASS on the pre-calibration constants (A10 1 percent, H100 0-3
  percent across three providers). CPU was 62 percent low ($0.40 vs
  $1.06 AWS r7i.4xlarge); A100 was 30 percent low vs hyperscaler
  prices though matched Runpod SXM within 10 percent.
- 6 measured llama.cpp single-stream decode speeds for Llama 3 8B and
  70B on A100 SXM 80GB and H100 PCIe 80GB (snapshot May 2024,
  llama-bench `tg1024`). **None passed the 15 percent gate**; the
  simulator was 8.4x to 31.1x too fast (median 17.7x). This matches
  the direction and magnitude of the three-anchor finding but is a
  stronger measurement because it uses a documented benchmark harness
  rather than published latency anecdotes.
- 8 Qwen2.5-Instruct MMLU-redux and HumanEval scores for 3B, 7B, 14B,
  and 72B. 6 of 8 within the 15 percent gate; small-model code quality
  underrated by 27-40 percent.
- 4 Qwen2 GPTQ-Int8/Int4 MMLU drops. INT8 penalty matched (0.01 vs
  measured 0.010-0.014); INT4 penalty too harsh (0.04 vs measured
  0.015-0.027).

### 6.5 Refit and verification

The refit replaces the originally-stated constants in
`infrarl/simulator/hardware.py`:

- `cpu`: `cost_per_hr_usd 0.40 -> 1.06`, `tps_13b_fp16 15 -> 0.92`
- `a10`: `cost_per_hr_usd 1.00` (unchanged), `tps_13b_fp16 180 -> 11.0`
  (speed scaled from A100 ratio; unverified by direct measurement)
- `a100`: `cost_per_hr_usd 2.40 -> 2.69`, `tps_13b_fp16 500 -> 30.5`
  (least-squares fit to Llama 3 8B/70B measurements)
- `h100`: `cost_per_hr_usd 4.50` (unchanged), `tps_13b_fp16 1100 -> 34.3`

`infrarl/simulator/models.py` chatbot and code-generation quality rows
are now sourced from Qwen2.5-Instruct benchmarks; the remaining four
use-case rows (batch_inference, rag, agentic, structured_extraction)
retain their pre-recalibration values because no matched public
benchmark exists for those tasks at the simulator's use-case granularity.
Precision penalties refit: `INT4` quality drop `0.04 -> 0.021`, `INT4`
speed factor `2.2 -> 3.08` (measured Q4_K_M vs F16 ratio), `INT8`
speed factor `1.6 -> 1.76` (geometric midpoint, not measured).

After refit the audit reports: price 5/11 within gate (remaining misses
reflect provider disagreement > 30 percent), decode speed 2/6 within
gate (simulator now 0.7x to 1.4x of measured), quality 8/8 within
gate, quantization penalty 4/4 within gate. The full `pytest -q tests`
suite passes 79/79; `python -m infrarl.evaluation.verify` reports zero
failures across the five gates.

### 6.6 Interpretation

The two-pass audit (three deep anchors plus 29 broad references)
agrees on the direction and refines the magnitude of the simulator's
latency error. The pre-refit simulator was adequate for comparing
policy designs under a shared trade-off model (the ranking in Table
2A is reproducible and internally consistent) but overestimated
single-stream throughput by roughly an order of magnitude, which means
it also overestimated the fraction of customer profiles any policy
could serve within the latency SLA.

The refit constants (now live in `main`) produce Table 2B in Section
4. The pivot from Table 2A to Table 2B is driven entirely by the
calibration; no agent code was retuned. Section 7's sensitivity
analysis predicted this pivot: a multiplicative perturbation of 10x
on latency sits well outside the plus-or-minus 20 percent envelope
under which Section 7 established ranking stability. Section 8
develops the implied methodology.

Sources for the three-anchor audit are at
`data/benchmarks/calibration_references.csv`; the gate output is at
`results/calibration_v1/`. Sources for the 29-reference recalibration
are at `data/benchmarks/external_references_20261127.csv` with
`docs/EXTERNAL_REFERENCE_CHECK.md` as the readable audit. The
recalibrated simulator's full re-run artifacts are at
`results/recalibrated_20261127/`.
