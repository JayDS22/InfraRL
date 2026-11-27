"""InfraRL live demo server.

    python -m webdemo.pretrain      # once (~10 min): trains and saves the demo agents
    python -m webdemo.server        # then open http://localhost:8501

Everything shown is computed live: the saved agents choose a stack step by step in
InfraEnv, and the simulator scores it. No numbers are precomputed or hardcoded.
"""
from __future__ import annotations

import itertools
import pickle
import threading
from dataclasses import asdict
from pathlib import Path

import numpy as np
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from infrarl.simulator.environment import (
    ARCH_RANGE, HW_RANGE, MODEL_RANGE, PREC_RANGE, SCALE_RANGE,
    STEP_ARCH, STEP_HW, STEP_MODEL, STEP_PREC, InfraEnv, terminal_reward,
)
from infrarl.simulator.estimator import SCALING_POLICIES, Stack, Tier, estimate, is_feasible
from infrarl.simulator.hardware import HARDWARE, HARDWARE_NAMES
from infrarl.simulator.models import MODEL_NAMES, PRECISION_NAMES, USE_CASES
from infrarl.simulator.profiles import SENSITIVITY, Profile, generate_profiles

HERE = Path(__file__).resolve().parent
TEST_SEED = 1000  # held-out test profiles used by the evaluation harness

AGENTS = [  # (key, display name, one-line description); order = display order
    ("lagrangian_ppo", "Lagrangian PPO", "Constrained RL, our best model"),
    ("ppo", "PPO", "Policy-gradient RL"),
    ("heuristic", "Heuristic", "Hand-written rules"),
    ("bandit_contextual", "LinUCB bandit", "Contextual bandit"),
    ("random", "Random", "Picks any valid stack"),
]

PRESETS = [
    {"id": "startup_chat", "name": "Startup chatbot", "blurb": "Customer-support bot, tight budget, fast replies",
     "profile": dict(use_case="chatbot", monthly_queries=2_000_000, latency_sla_ms=1200, acc_req=0.78,
                     monthly_budget_usd=1500, data_sensitivity="low", peak_multiplier=3.0,
                     avg_input_tokens=300, avg_output_tokens=200, concurrent_users=300)},
    {"id": "code_assist", "name": "Code assistant", "blurb": "IDE copilot, needs high accuracy",
     "profile": dict(use_case="code_generation", monthly_queries=800_000, latency_sla_ms=3000, acc_req=0.85,
                     monthly_budget_usd=12000, data_sensitivity="high", peak_multiplier=2.5,
                     avg_input_tokens=1200, avg_output_tokens=350, concurrent_users=400)},
    {"id": "batch", "name": "Nightly batch job", "blurb": "Millions of documents, latency barely matters",
     "profile": dict(use_case="batch_inference", monthly_queries=20_000_000, latency_sla_ms=20000, acc_req=0.70,
                     monthly_budget_usd=6000, data_sensitivity="medium", peak_multiplier=1.5,
                     avg_input_tokens=500, avg_output_tokens=120, concurrent_users=50)},
    {"id": "rag", "name": "Enterprise search (RAG)", "blurb": "Long documents in, short answers out",
     "profile": dict(use_case="rag", monthly_queries=3_000_000, latency_sla_ms=1500, acc_req=0.80,
                     monthly_budget_usd=8000, data_sensitivity="high", peak_multiplier=4.0,
                     avg_input_tokens=3000, avg_output_tokens=200, concurrent_users=1000)},
    {"id": "agentic", "name": "AI agent platform", "blurb": "Multi-step reasoning, hardest quality bar",
     "profile": dict(use_case="agentic", monthly_queries=500_000, latency_sla_ms=6000, acc_req=0.85,
                     monthly_budget_usd=15000, data_sensitivity="medium", peak_multiplier=3.0,
                     avg_input_tokens=2000, avg_output_tokens=500, concurrent_users=200)},
]

HW_LABEL = {"cpu": "CPU", "a10": "A10 GPU", "a100": "A100 GPU", "h100": "H100 GPU"}
ARCH_LABEL = {1: "Single tier", 2: "Two-tier router", 3: "Three-tier router"}
SCALE_LABEL = {"static": "Static (peak)", "conservative": "Autoscale, safe", "aggressive": "Autoscale, lean"}
STEP_NAME = {STEP_ARCH: "Architecture", STEP_HW: "Hardware", STEP_MODEL: "Model", STEP_PREC: "Precision"}

_lock = threading.Lock()
_agents: dict[str, object] = {}
_test_profiles = generate_profiles(1000, seed=TEST_SEED)
_env = InfraEnv(_test_profiles)


def load_agents() -> None:
    for key, *_ in AGENTS:
        path = HERE / "models" / f"{key}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                _agents[key] = pickle.load(f)


def _choice_label(action: int) -> str:
    if action in ARCH_RANGE:
        return ARCH_LABEL[action - ARCH_RANGE.start + 1]
    if action in HW_RANGE:
        return HW_LABEL[HARDWARE_NAMES[action - HW_RANGE.start]]
    if action in MODEL_RANGE:
        return MODEL_NAMES[action - MODEL_RANGE.start].upper() + " model"
    if action in PREC_RANGE:
        return PRECISION_NAMES[action - PREC_RANGE.start].upper()
    return SCALE_LABEL[SCALING_POLICIES[action - SCALE_RANGE.start]]


def _scores(stack: Stack, q: float, lat: float, cost: float, profile: Profile) -> dict:
    reward, parts = terminal_reward(q, lat, cost, profile)
    return {
        "quality": q, "latency_ms": lat, "cost_usd": cost, "reward": reward,
        "meets_quality": parts["meets_quality"], "meets_sla": parts["meets_sla"],
        "within_budget": parts["within_budget"],
        "meets_all": parts["meets_quality"] and parts["meets_sla"] and parts["within_budget"],
        "tiers": [{"hardware": HW_LABEL[t.hardware], "model": t.model.upper(), "precision": t.precision.upper()}
                  for t in stack.tiers],
        "scaling": SCALE_LABEL[stack.scaling],
    }


def run_agent(key: str, profile: Profile) -> dict:
    """Greedy episode for one agent; records every decision it makes."""
    agent = _agents[key]
    env = _env
    obs, info = env.reset(options={"profile": profile})
    agent.begin_episode(profile)
    mask, decisions = info["action_mask"], []
    while True:
        step_type, tier_idx = env._schedule[env._t]
        a = int(agent.select_action(obs, mask, explore=False))
        label = STEP_NAME.get(step_type, "Scaling")
        if step_type in (STEP_HW, STEP_MODEL, STEP_PREC) and env.tier_count > 1:
            label += f" · tier {tier_idx + 1}"
        decisions.append({"step": label, "choice": _choice_label(a), "options": int(mask.sum())})
        obs, _, term, _, info = env.step(a)
        mask = info["action_mask"]
        if term:
            break
    s = info["stack"]
    return {"decisions": decisions, **_scores(s, info["quality"], info["latency_ms"], info["cost_usd"], profile)}


_FEASIBLE_TIERS = [Tier(h, m, p) for h, m, p in itertools.product(HARDWARE_NAMES, MODEL_NAMES, PRECISION_NAMES)
                   if is_feasible(Tier(h, m, p))]


def oracle(profile: Profile) -> dict:
    """Exhaustive search over every 1- and 2-tier stack x scaling policy (the yardstick)."""
    best, best_r, n = None, -1.0, 0
    for k in (1, 2):
        for tiers in itertools.product(_FEASIBLE_TIERS, repeat=k):
            for sc in SCALING_POLICIES:
                st = Stack(tuple(tiers), sc)
                q, lat, cost = estimate(st, profile)
                r, _ = terminal_reward(q, lat, cost, profile)
                n += 1
                if r > best_r:
                    best, best_r = (st, q, lat, cost), r
    st, q, lat, cost = best
    return {"searched": n, **_scores(st, q, lat, cost, profile)}


def _profile_from(d: dict) -> Profile:
    return Profile(
        use_case=d["use_case"], monthly_queries=float(d["monthly_queries"]),
        latency_sla_ms=float(d["latency_sla_ms"]), acc_req=float(d["acc_req"]),
        monthly_budget_usd=float(d["monthly_budget_usd"]), data_sensitivity=d.get("data_sensitivity", "medium"),
        peak_multiplier=float(d["peak_multiplier"]), avg_input_tokens=float(d["avg_input_tokens"]),
        avg_output_tokens=float(d["avg_output_tokens"]), concurrent_users=float(d["concurrent_users"]),
    )


async def meta(_: Request) -> JSONResponse:
    return JSONResponse({
        "agents": [{"key": k, "name": n, "desc": d, "loaded": k in _agents} for k, n, d in AGENTS],
        "use_cases": list(USE_CASES), "sensitivity": list(SENSITIVITY), "presets": PRESETS,
    })


async def recommend(request: Request) -> JSONResponse:
    profile = _profile_from(await request.json())
    with _lock:
        results = {k: run_agent(k, profile) for k, *_ in AGENTS if k in _agents}
        best = oracle(profile)
    return JSONResponse({"profile": asdict(profile), "results": results, "oracle": best})


async def random_customer(request: Request) -> JSONResponse:
    rng = np.random.default_rng(int(request.query_params.get("seed", np.random.SeedSequence().entropy % 2**32)))
    i = int(rng.integers(len(_test_profiles)))
    return JSONResponse({"index": i, "profile": asdict(_test_profiles[i])})


async def battle(request: Request) -> JSONResponse:
    """All agents on n unseen held-out test customers (the same set the evaluation uses)."""
    body = await request.json()
    n = max(1, min(int(body.get("n", 50)), 300))
    rng = np.random.default_rng(int(body.get("seed", 0)))
    idx = rng.choice(len(_test_profiles), size=n, replace=False)
    rows = []
    with _lock:
        for i in idx:
            p = _test_profiles[int(i)]
            row = {"index": int(i), "use_case": p.use_case, "acc_req": p.acc_req,
                   "budget": p.monthly_budget_usd, "sla_ms": p.latency_sla_ms, "agents": {}}
            for k, *_ in AGENTS:
                if k in _agents:
                    r = run_agent(k, p)
                    row["agents"][k] = {kk: r[kk] for kk in ("meets_all", "meets_quality", "meets_sla",
                                                             "within_budget", "cost_usd", "reward")}
            rows.append(row)
    return JSONResponse({"n": n, "rows": rows})


async def index(_: Request) -> FileResponse:
    return FileResponse(HERE / "static" / "index.html")


load_agents()
app = Starlette(routes=[
    Route("/", index),
    Route("/api/meta", meta),
    Route("/api/recommend", recommend, methods=["POST"]),
    Route("/api/random_customer", random_customer),
    Route("/api/battle", battle, methods=["POST"]),
    Mount("/static", StaticFiles(directory=HERE / "static"), name="static"),
])

if __name__ == "__main__":
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8501)
    args = ap.parse_args()
    missing = [k for k, *_ in AGENTS if k not in _agents]
    if missing:
        print(f"[webdemo] not trained yet (run python -m webdemo.pretrain): {missing}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
