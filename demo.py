"""Simulator smoke test: hardcode one profile and one stack, print one output.

Not the full simulator yet. Just enough to show the (Q, L, cost) pipeline
is wired. Numbers are rough estimates from Nebius Token Factory published specs.
"""

HARDWARE = {
    "a100": {"memory_gb": 80, "cost_per_hr_usd": 2.40, "tokens_per_sec_13b_fp16": 500},
}

MODELS = {
    "13b": {"min_memory_fp16_gb": 26, "quality_chatbot": 0.85},
}


def estimate(hw_name, model_name, profile):
    hw = HARDWARE[hw_name]
    m = MODELS[model_name]
    quality = m["quality_chatbot"]
    latency_ms = (profile["avg_output_tokens"] / hw["tokens_per_sec_13b_fp16"]) * 1000
    monthly_cost = hw["cost_per_hr_usd"] * 24 * 30
    return quality, latency_ms, monthly_cost


if __name__ == "__main__":
    profile = {
        "use_case": "chatbot",
        "avg_output_tokens": 200,
        "monthly_budget_usd": 2000,
        "latency_sla_ms": 500,
        "acc_req": 0.80,
    }
    stack = ("a100", "13b")
    q, l, c = estimate(*stack, profile)

    print(f"profile:  {profile['use_case']}")
    print(f"stack:    {stack}")
    print(f"quality:  {q:.2f}  (req {profile['acc_req']})")
    print(f"latency:  {l:.0f} ms  (SLA {profile['latency_sla_ms']} ms)")
    print(f"cost:     ${c:.0f}/mo  (budget ${profile['monthly_budget_usd']})")
