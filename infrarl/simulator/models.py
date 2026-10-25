"""Model / precision lookup tables.

Quality is per use case, calibrated against public benchmarks (LMSYS Arena ELO
normalised to [0, 1] for chatbot, HumanEval pass@1 for code generation, MMLU
for structured extraction / RAG / batch). Values below are placeholders until
the calibration pass is done.
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
PRECISION_SPEED = {"int4": 2.2, "int8": 1.6, "fp16": 1.0, "fp32": 0.5}
# absolute quality drop vs fp16
PRECISION_QUALITY_PENALTY = {"int4": 0.04, "int8": 0.01, "fp16": 0.0, "fp32": 0.0}
# weights + KV cache / activation overhead
MEMORY_OVERHEAD = 1.0

# quality at fp16, indexed by model size
BASE_QUALITY = {
    "chatbot": {"3b": 0.62, "7b": 0.76, "13b": 0.85, "70b": 0.95},
    "code_generation": {"3b": 0.45, "7b": 0.62, "13b": 0.74, "70b": 0.92},
    "batch_inference": {"3b": 0.60, "7b": 0.72, "13b": 0.80, "70b": 0.90},
    "rag": {"3b": 0.58, "7b": 0.70, "13b": 0.80, "70b": 0.91},
    "agentic": {"3b": 0.40, "7b": 0.58, "13b": 0.72, "70b": 0.90},
    "structured_extraction": {"3b": 0.65, "7b": 0.77, "13b": 0.85, "70b": 0.93},
}


def min_memory_gb(model: str, precision: str) -> float:
    """Minimum accelerator memory needed to host one replica."""
    return PARAMS_B[model] * BYTES_PER_PARAM[precision] * MEMORY_OVERHEAD
