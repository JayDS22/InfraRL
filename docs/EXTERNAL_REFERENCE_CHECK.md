# External reference check of simulator constants (2026-11-27)

Answers the interim-report deduction that "all results rely on placeholder simulator
constants". Every reference below is a real, publicly published number fetched on
2026-11-27; nothing is estimated. Raw rows with URLs: `data/benchmarks/external_references_20261127.csv`.
Reproduce: `python scripts/external_reference_check.py`.

**What is checked:** the simulator's primitive constants (GPU/CPU hourly price,
single-stream decode tokens/s, per-size quality, quantization penalty), read live from
`infrarl/simulator/*.py`. Gate: within 15% (interim report's V10 gate); quantization
penalty uses +/-0.01 absolute.

## Results

| Axis | Simulator key | Simulator | Real (sourced) | Units | Error | Gate | Source |
|---|---|---|---|---|---|---|---|
| price | `cpu` | 0.4 | 1.06 | USD/hr | -62% | FAIL | AWS EC2 r7i.4xlarge (16 vCPU 128 GiB) on-demand via Vantage |
| price | `a10` | 1 | 1.01 | USD/hr | -1% | PASS | AWS EC2 g5.xlarge (1x A10G 24 GiB) on-demand via Vantage |
| price | `a10` | 1 | 1.29 | USD/hr | -22% | FAIL | Lambda 1x NVIDIA A10 24 GB |
| price | `a100` | 2.4 | 3.43 | USD/hr | -30% | FAIL | AWS EC2 p4de.24xlarge (8x A100 80 GB) on-demand via Vantage |
| price | `a100` | 2.4 | 3.19 | USD/hr | -25% | FAIL | Lambda NVIDIA HGX A100 80 GB |
| price | `a100` | 2.4 | 2.19 | USD/hr | +10% | PASS | Runpod A100 SXM 80 GB (Secure Cloud pod) |
| price | `a100` | 2.4 | 2.09 | USD/hr | +15% | PASS | Runpod A100 PCIe 80 GB (Secure Cloud pod) |
| price | `h100` | 4.5 | 6.88 | USD/hr | -35% | FAIL | AWS EC2 p5.48xlarge (8x H100 80 GB) on-demand via Vantage |
| price | `h100` | 4.5 | 4.5 | USD/hr | +0% | PASS | Nebius NVIDIA HGX H100 on-demand |
| price | `h100` | 4.5 | 4.59 | USD/hr | -2% | PASS | Lambda NVIDIA HGX H100 80 GB |
| price | `h100` | 4.5 | 4.39 | USD/hr | +3% | PASS | Runpod H100 SXM 80 GB (Secure Cloud pod) |
| decode_tps | `a100|7b|fp16` | 929 | 53.2 | tokens/s | +1646% | FAIL | llama.cpp llama-bench tg1024 Llama 3 8B F16 on A100 SXM 80GB |
| decode_tps | `a100|7b|int4` | 2.04e+03 | 133 | tokens/s | +1432% | FAIL | llama.cpp llama-bench tg1024 Llama 3 8B Q4_K_M on A100 SXM 80GB |
| decode_tps | `a100|70b|int4` | 204 | 24.3 | tokens/s | +740% | FAIL | llama.cpp llama-bench tg1024 Llama 3 70B Q4_K_M on A100 SXM 80GB |
| decode_tps | `h100|7b|fp16` | 2.04e+03 | 67.8 | tokens/s | +2914% | FAIL | llama.cpp llama-bench tg1024 Llama 3 8B F16 on H100 PCIe 80GB |
| decode_tps | `h100|7b|int4` | 4.49e+03 | 144 | tokens/s | +3010% | FAIL | llama.cpp llama-bench tg1024 Llama 3 8B Q4_K_M on H100 PCIe 80GB |
| decode_tps | `h100|70b|int4` | 449 | 25 | tokens/s | +1697% | FAIL | llama.cpp llama-bench tg1024 Llama 3 70B Q4_K_M on H100 PCIe 80GB |
| quality | `chatbot|3b` | 0.62 | 0.644 | fraction | -4% | PASS | Qwen2.5-3B-Instruct MMLU-redux |
| quality | `chatbot|7b` | 0.76 | 0.754 | fraction | +1% | PASS | Qwen2.5-7B-Instruct MMLU-redux |
| quality | `chatbot|13b` | 0.85 | 0.8 | fraction | +6% | PASS | Qwen2.5-14B-Instruct MMLU-redux |
| quality | `chatbot|70b` | 0.95 | 0.868 | fraction | +9% | PASS | Qwen2.5-72B-Instruct MMLU-redux |
| quality | `code_generation|3b` | 0.45 | 0.744 | fraction | -40% | FAIL | Qwen2.5-3B-Instruct HumanEval |
| quality | `code_generation|7b` | 0.62 | 0.848 | fraction | -27% | FAIL | Qwen2.5-7B-Instruct HumanEval |
| quality | `code_generation|13b` | 0.74 | 0.835 | fraction | -11% | PASS | Qwen2.5-14B-Instruct HumanEval |
| quality | `code_generation|70b` | 0.92 | 0.866 | fraction | +6% | PASS | Qwen2.5-72B-Instruct HumanEval |
| quant_drop | `7b|int8` | 0.01 | 0.014 | fraction | -0.004 abs | PASS | Qwen2-7B-Instruct MMLU BF16 70.5 -> GPTQ-Int8 69.1 |
| quant_drop | `7b|int4` | 0.04 | 0.027 | fraction | +0.013 abs | FAIL | Qwen2-7B-Instruct MMLU BF16 70.5 -> GPTQ-Int4 67.8 |
| quant_drop | `70b|int8` | 0.01 | 0.01 | fraction | +0.000 abs | PASS | Qwen2-72B-Instruct MMLU BF16 82.3 -> GPTQ-Int8 81.3 |
| quant_drop | `70b|int4` | 0.04 | 0.015 | fraction | +0.025 abs | FAIL | Qwen2-72B-Instruct MMLU BF16 82.3 -> GPTQ-Int4 80.8 |

- **price**: 6/11 references within gate
- **decode_tps**: 0/6 references within gate
- **quality**: 6/8 references within gate
- **quant_drop**: 2/4 references within gate
- decode speed: simulator is 8.4x to 31.1x faster than measured (median 17.7x)

## What this means

1. **Prices are roughly right.** H100 matches Nebius, Lambda and Runpod (within 3%); A10 matches
   AWS (1%). A100 ($2.40) sits between marketplace (Runpod $2.09-2.19) and hyperscaler/Lambda
   ($3.19-3.43) prices. **CPU is too cheap**: $0.40/hr vs $1.06/hr for a 128 GiB AWS instance.
2. **Generation speed is far too optimistic.** The simulator's single-stream decode speed is
   8x-31x faster than measured (median ~18x). This is not a benchmark-tool artifact: decoding is
   memory-bandwidth bound, and an 8B FP16 model (16 GB of weights) on an A100 (about 2 TB/s)
   cannot exceed roughly 125 tokens/s, while the simulator assumes 929. Consequence: simulated
   **latency is about an order of magnitude too low**, so SLA compliance in all results is
   inflated.
3. **Quality is reasonable for general tasks, off for small code models.** Chatbot quality is
   within 9% of MMLU-redux for all four sizes. Code generation underrates small models (3B: 0.45
   vs 0.744 HumanEval) because modern small instruct models are much better at code than the table assumes.
4. **INT8 penalty matches; INT4 penalty is too harsh** (0.04 vs measured 0.015-0.027).

## Mapping decisions (state these in the report)
- Size buckets 3b/7b/13b/70b map to Qwen2.5-Instruct 3B/7B/14B/72B (the one public family
  covering all four sizes) for quality, and to Llama 3 8B/70B for speed (the only sizes in the
  speed benchmark). 13b -> 14B and 70b -> 72B are approximations.
- Chatbot quality is proxied by MMLU-redux and code generation by HumanEval pass@1. RAG,
  agentic, batch and structured-extraction quality have no matched public benchmark and are not checked.
- Speed references are llama.cpp batch-1 `tg1024` (May 2024 snapshot); Q4_K_M (~4.8 bit) is
  treated as INT4. H100 speed is the PCIe variant.
- Per-GPU prices from 8-GPU AWS/Lambda instances are the instance price divided by 8.

## Not covered (no public measured reference found)
- A10 and server-CPU decode speed: no published single-stream measurement found. A local CPU
  measurement (llama.cpp on a laptop) or a vLLM run on Zaratan A100 40 GB would fill this.
- End-to-end latency under load and multi-tier routing behaviour.

## After recalibration (same check, new constants)

The tables above are the check against the original placeholder constants. After updating
`infrarl/simulator/hardware.py` and `models.py` from these references (fit rules are in
those files' docstrings), `python scripts/external_reference_check.py` gives:

- price: 5/11 within 15% (remaining misses are spread between providers: AWS vs Runpod differ by >30%)
- decode speed: 2/6 within 15%; simulator now 0.7x-1.4x of measured (was 8x-31x)
- quality: 8/8 within 15%
- quantization penalty: 4/4 within +/-0.01

Re-run results with the recalibrated simulator: `results/recalibrated_20261127/RESULTS_SUMMARY.md`.
Note: quality and speed were fitted to these same references, so this is a calibration
fit, not an independent holdout test.
