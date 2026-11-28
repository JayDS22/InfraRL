"""Populate data/benchmarks/calibration_references.csv with sourced public
references for the three primary calibration anchors (cpu_3b_int8,
a10_13b_int8, a100_70b_int8).

Numbers were retrieved from publicly-documented sources on 2026-11-26.
Each cell carries its source URL, publication date, and matched-vs-mismatched
measurement_type disposition. Quality cells are intentionally marked with a
measurement_type that does NOT match the simulator's quality head
(use_case_fraction), so `calibration.comparable()` returns NOT_COMPARABLE
rather than scoring an apples-to-oranges residual. Latency and cost cells
use matched measurement_type and will go through the residual gate.
"""
from __future__ import annotations

import csv
from pathlib import Path

CSV_PATH = Path("data/benchmarks/calibration_references.csv")

CONFIG_COMMON = {
    "cpu_3b_int8": {
        "provider": "aws",
        "region": "us-east-1",
        "hardware_id": "cpu",
        "sim_hardware_id": "cpu",
        "gpu_vram_gb": "",
        "gpu_count": "1",
        "model_id": "3b",
        "sim_model_id": "3b",
        "quantizer": "Q8_0",
        "runtime": "llama.cpp",
        "batch": "1",
        "concurrency": "1",
        "input_tokens": "500",
        "output_tokens": "150",
        "statistic": "mean",
        "workload": "single_stream_chatbot",
        "use_case": "chatbot",
        "data_sensitivity": "medium",
        "monthly_queries": "100000",
        "peak_multiplier": "2.0",
        "avg_input_tokens": "500",
        "avg_output_tokens": "150",
        "concurrent_users": "1",
        "latency_sla_ms": "10000",
        "acc_req": "0.5",
        "monthly_budget_usd": "10000",
        "scaling": "static",
        "retrieved_utc": "2026-11-26",
    },
    "a10_13b_int8": {
        "provider": "aws",
        "region": "us-east-1",
        "hardware_id": "a10",
        "sim_hardware_id": "a10",
        "gpu_vram_gb": "24",
        "gpu_count": "1",
        "model_id": "13b",
        "sim_model_id": "13b",
        "quantizer": "INT8",
        "runtime": "vLLM",
        "batch": "1",
        "concurrency": "1",
        "input_tokens": "500",
        "output_tokens": "150",
        "statistic": "mean",
        "workload": "single_stream_chatbot",
        "use_case": "chatbot",
        "data_sensitivity": "medium",
        "monthly_queries": "100000",
        "peak_multiplier": "2.0",
        "avg_input_tokens": "500",
        "avg_output_tokens": "150",
        "concurrent_users": "1",
        "latency_sla_ms": "10000",
        "acc_req": "0.5",
        "monthly_budget_usd": "10000",
        "scaling": "static",
        "retrieved_utc": "2026-11-26",
    },
    "a100_70b_int8": {
        "provider": "aws",
        "region": "us-east-1",
        "hardware_id": "a100",
        "sim_hardware_id": "a100",
        "gpu_vram_gb": "80",
        "gpu_count": "1",
        "model_id": "70b",
        "sim_model_id": "70b",
        "quantizer": "AWQ_INT4",
        "runtime": "vLLM",
        "batch": "1",
        "concurrency": "1",
        "input_tokens": "500",
        "output_tokens": "150",
        "statistic": "mean",
        "workload": "single_stream_chatbot",
        "use_case": "chatbot",
        "data_sensitivity": "medium",
        "monthly_queries": "100000",
        "peak_multiplier": "2.0",
        "avg_input_tokens": "500",
        "avg_output_tokens": "150",
        "concurrent_users": "1",
        "latency_sla_ms": "10000",
        "acc_req": "0.5",
        "monthly_budget_usd": "10000",
        "scaling": "static",
        "retrieved_utc": "2026-11-26",
    },
}

CELL_REFS = {
    ("cpu_3b_int8", "quality"): {
        "reference": "0.634",
        "source_url": "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct",
        "publication_date": "2024-09-25",
        "effective_date": "2024-09-25",
        "measurement_type": "MMLU_5shot_frac",
        "sim_measurement_type": "use_case_fraction",
    },
    ("cpu_3b_int8", "latency"): {
        "reference": "8000",
        "source_url": "https://blog.victorbona.dev/compendium/cpu-llm-inference/benchmarks-and-baselines",
        "publication_date": "2025-06-01",
        "effective_date": "2025-06-01",
        "measurement_type": "e2e_mean_ms",
        "sim_measurement_type": "e2e_mean_ms",
    },
    ("cpu_3b_int8", "cost"): {
        "reference": "2056.32",
        "source_url": "https://www.economize.cloud/resources/aws/pricing/ec2/c7i.16xlarge/",
        "publication_date": "2026-11-01",
        "effective_date": "2026-11-01",
        "measurement_type": "monthly_usd",
        "sim_measurement_type": "monthly_usd",
    },
    ("a10_13b_int8", "quality"): {
        "reference": "0.546",
        "source_url": "https://arxiv.org/abs/2307.09288",
        "publication_date": "2023-07-18",
        "effective_date": "2023-07-18",
        "measurement_type": "MMLU_5shot_frac",
        "sim_measurement_type": "use_case_fraction",
    },
    ("a10_13b_int8", "latency"): {
        "reference": "2586",
        "source_url": "https://markaicode.com/benchmarks/vllm-throughput-benchmark/",
        "publication_date": "2024-12-01",
        "effective_date": "2024-12-01",
        "measurement_type": "e2e_mean_ms",
        "sim_measurement_type": "e2e_mean_ms",
    },
    ("a10_13b_int8", "cost"): {
        "reference": "724.32",
        "source_url": "https://ec2calc.cloud/instances/g5.xlarge",
        "publication_date": "2026-11-01",
        "effective_date": "2026-11-01",
        "measurement_type": "monthly_usd",
        "sim_measurement_type": "monthly_usd",
    },
    ("a100_70b_int8", "quality"): {
        "reference": "0.836",
        "source_url": "https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct",
        "publication_date": "2024-07-23",
        "effective_date": "2024-07-23",
        "measurement_type": "MMLU_5shot_frac",
        "sim_measurement_type": "use_case_fraction",
    },
    ("a100_70b_int8", "latency"): {
        "reference": "4286",
        "source_url": "https://www.spheron.network/blog/awq-quantization-guide-llm-deployment/",
        "publication_date": "2026-01-01",
        "effective_date": "2026-01-01",
        "measurement_type": "e2e_mean_ms",
        "sim_measurement_type": "e2e_mean_ms",
    },
    ("a100_70b_int8", "cost"): {
        "reference": "2470.14",
        "source_url": "https://www.devzero.io/instances/aws/p4de.24xlarge",
        "publication_date": "2026-11-01",
        "effective_date": "2026-11-01",
        "measurement_type": "monthly_usd",
        "sim_measurement_type": "monthly_usd",
    },
}


def main() -> None:
    with CSV_PATH.open() as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)

    touched = 0
    for row in rows:
        if row["role"] != "calibration":
            continue
        config_id = row["config_id"]
        axis = row["axis"]
        if (config_id, axis) not in CELL_REFS:
            continue
        for k, v in CONFIG_COMMON[config_id].items():
            if k in row:
                row[k] = v
        for k, v in CELL_REFS[(config_id, axis)].items():
            if k in row:
                row[k] = v
        row["reference_status"] = "MATCHED_REFERENCE"
        row["independent_reference"] = "true"
        touched += 1

    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"updated {touched} calibration rows in {CSV_PATH}")


if __name__ == "__main__":
    main()
