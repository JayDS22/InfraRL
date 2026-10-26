"""Streamlit demo: enter a customer profile, get a recommended stack from each method.

    streamlit run infrarl/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from infrarl.agents.registry import REGISTRY
from infrarl.evaluation.experiments import run_episode, train
from infrarl.simulator.environment import InfraEnv
from infrarl.simulator.models import USE_CASES
from infrarl.simulator.profiles import SENSITIVITY, Profile, generate_profiles

st.set_page_config(page_title="InfraRL", layout="wide")
st.title("InfraRL: AI infrastructure recommender")
st.caption("Cold-start stack recommendation (architecture, hardware, model, precision, scaling) "
           "scored by a calibrated lookup-table simulator. Numbers are simulated, not live benchmarks.")

with st.sidebar:
    st.header("Customer profile")
    use_case = st.selectbox("Use case", USE_CASES)
    queries = st.number_input("Monthly queries", 10_000, 100_000_000, 2_000_000, step=100_000)
    sla = st.number_input("Latency SLA (ms)", 100, 60_000, 1000, step=100)
    acc = st.slider("Accuracy requirement", 0.5, 0.95, 0.80, 0.01)
    budget = st.number_input("Monthly budget (USD)", 500, 200_000, 5000, step=500)
    sens = st.selectbox("Data sensitivity", SENSITIVITY)
    peak = st.slider("Peak-traffic multiplier", 1.0, 6.0, 3.0, 0.5)
    tok_in = st.number_input("Avg input tokens", 10, 20_000, 500, step=50)
    tok_out = st.number_input("Avg output tokens", 10, 5_000, 200, step=10)
    users = st.number_input("Concurrent users", 1, 100_000, 200, step=10)
    agents = st.multiselect("Methods", list(REGISTRY), default=["heuristic", "bandit_contextual", "random"])

profile = Profile(use_case, float(queries), float(sla), float(acc), float(budget), sens, float(peak),
                  float(tok_in), float(tok_out), float(users))


@st.cache_resource(show_spinner="Training agent on 1000 synthetic profiles...")
def trained_agent(name: str):
    spec = REGISTRY[name]
    profiles = generate_profiles(1000, seed=2000)
    env = InfraEnv(profiles)
    agent = spec.factory(0)
    if spec.train_epochs:
        train(env, agent, profiles, spec.train_epochs, seed=0)
    return agent


rows = []
for name in agents:
    env = InfraEnv([profile])
    r = run_episode(env, trained_agent(name), profile, train=False)
    rows.append({
        "method": name, "stack": r.stack.replace("|", " | "), "quality": round(r.quality, 3),
        "latency_ms": round(r.latency_ms), "cost_usd/mo": round(r.cost_usd),
        "quality ok": r.meets_quality, "budget ok": r.within_budget, "SLA ok": r.meets_sla,
        "reward": round(r.reward, 3),
    })

if rows:
    df = pd.DataFrame(rows)
    best = df.sort_values("reward", ascending=False).iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best method (by reward)", best["method"])
    c2.metric("Quality", best["quality"], delta=f"req {acc:.2f}")
    c3.metric("Latency (ms)", best["latency_ms"], delta=f"SLA {sla}", delta_color="off")
    c4.metric("Cost / month", f"${best['cost_usd/mo']:,}", delta=f"budget ${budget:,}", delta_color="off")
    st.dataframe(df, width="stretch", hide_index=True)
else:
    st.info("Pick at least one method in the sidebar.")
