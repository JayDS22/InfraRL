"""Model / precision lookup tables, partly recalibrated 2026-11-27 (see docs/EXTERNAL_REFERENCE_CHECK.md).

Size buckets map to Qwen2.5-Instruct 3B/7B/14B/72B. chatbot = MMLU-redux;
code_generation = mean of HumanEval and MBPP pass@1 (HumanEval alone is non-monotone
between 7B and 14B). The other four use cases have no matched public benchmark and keep
their original PLACEHOLDER values. Precision penalties: INT8 matches Qwen2 GPTQ-Int8
MMLU drops (0.010-0.014); INT4 = mean of Qwen2 7B/72B GPTQ-Int4 drops (0.027, 0.015).
Decode speed factors: INT4 fitted to measured llama.cpp Q4_K_M vs F16 (3.08); INT8 is the
geometric midpoint (1.76, not measured); FP32 = half of FP16 by weight bytes.
"""
from __future__ import annotations

MODEL_NAMES = ("3b", "7b", "13b", "70b")
PRECISION_NAMES = ("int4", "int8", "fp16", "fp32")
USE_CASES = (
    "chatbot",
    "code_generation",
    "batch_inference",
    "rag",
    "agentic",
    "structured_extraction",
)

PARAMS_B = {"3b": 3, "7b": 7, "13b": 13, "70b": 70}
BYTES_PER_PARAM = {"int4": 0.5, "int8": 1.0, "fp16": 2.0, "fp32": 4.0}
# decode speed relative to fp16
PRECISION_SPEED = {"int4": 3.08, "int8": 1.76, "fp16": 1.0, "fp32": 0.5}
# absolute quality drop vs fp16
PRECISION_QUALITY_PENALTY = {"int4": 0.021, "int8": 0.01, "fp16": 0.0, "fp32": 0.0}
# weights + KV cache / activation overhead
MEMORY_OVERHEAD = 1.0

# quality at fp16, indexed by model size
BASE_QUALITY = {
    "chatbot": {"3b": 0.644, "7b": 0.754, "13b": 0.800, "70b": 0.868},  # sourced
    "code_generation": {"3b": 0.736, "7b": 0.820, "13b": 0.828, "70b": 0.874},  # sourced
    "batch_inference": {"3b": 0.60, "7b": 0.72, "13b": 0.80, "70b": 0.90},
    "rag": {"3b": 0.58, "7b": 0.70, "13b": 0.80, "70b": 0.91},
    "agentic": {"3b": 0.40, "7b": 0.58, "13b": 0.72, "70b": 0.90},
    "structured_extraction": {"3b": 0.65, "7b": 0.77, "13b": 0.85, "70b": 0.93},
}


def min_memory_gb(model: str, precision: str) -> float:
    """Minimum accelerator memory needed to host one replica."""
    return PARAMS_B[model] * BYTES_PER_PARAM[precision] * MEMORY_OVERHEAD
