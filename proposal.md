# InfraRL: Reinforcement Learning for Use-Case-Driven AI Infrastructure Recommendation

## Team Members

- Anupama - Simulation and Data Engineering
- Jay - Reinforcement Learning Implementation
- Vishnu - Evaluation, Integration, and Reporting

---

## 1. Background

The rapid growth of LLM-based applications has created a complex infrastructure decision problem. Companies deploying AI systems must choose from a large space of options: GPU type (A10, A100, H100), model size (3B to 70B parameters), quantization precision (INT4 to FP32), batch strategies, and scaling policies. A wrong choice means either overspending on hardware or failing to meet latency and quality requirements.

Currently, this decision is made either by manual consultation with cloud sales engineers (expensive and slow) or by trial and error (wasteful). Cloud providers like Nebius, AWS, and Azure offer dozens of GPU configurations but no intelligent system that maps a customer's specific use case to an optimal infrastructure stack. The result is that most companies either overprovision (wasting 40-60% of compute budget) or underprovision (missing SLA targets).

This problem maps naturally to reinforcement learning. The infrastructure recommendation can be formulated as a sequential decision process: given a customer profile (state), select hardware, model, configuration, and scaling decisions (actions) that maximize a reward combining quality, cost efficiency, and SLA compliance.

---

## 2. Significance and Statement of the Problem

**Problem Statement:** Given a customer's AI use case profile (application type, query volume, latency SLA, accuracy requirements, budget, and data sensitivity), automatically recommend an optimal multi-layer infrastructure stack from high-level architecture design down to low-level hardware and model configuration, and continuously adapt the recommendation as workload patterns evolve.

**Why this matters:**

- The AI cloud infrastructure market is projected to exceed $300B by 2027. Every customer faces this decision.
- Manual infrastructure planning takes days to weeks and requires specialized expertise that most teams lack.
- Static recommendations become stale as workloads change over time. A production chatbot might see 3x traffic during business hours and need to scale down at night.
- No existing system uses RL to solve this problem. Model routing papers (RouteLLM, Lodestar) pick which model to use but ignore hardware selection, precision tuning, and scaling. GPU allocation papers (Temporal-Aware GPU Allocation) work at the cluster scheduling level, not the per-customer recommendation level.

**Our contribution:** We build the first RL-based system that jointly optimizes hardware selection, model choice, configuration parameters, and scaling policy for a given use case, under cost and latency constraints.

---

## 3. Approach

### 3.1 MDP Formulation

We formulate the recommendation as a Markov Decision Process:

**State space:** A customer profile vector encoding use case type, daily query volume, latency SLA, accuracy requirement, monthly budget, data sensitivity, peak traffic pattern, average input/output token counts, and concurrent users.

**Action space:** The agent makes sequential decisions across five layers:
1. Architecture type (single-tier, two-tier, or three-tier routing)
2. Hardware per tier (CPU, A10, A100, H100)
3. Model per tier (3B, 7B, 13B, 70B parameter models)
4. Configuration per tier (precision, batch size, context window, speculative decoding)
5. Scaling policy (static, conservative autoscale, aggressive autoscale)

Invalid combinations (e.g., 70B model on A10 with only 24GB memory) are masked from the action space.

**Reward function:** A weighted combination of:
- Quality score: simulated task accuracy for the recommended model on the given use case (weight: 0.4)
- Budget compliance: binary reward for staying within budget, with proportional bonus for cost savings (weight: 0.3)
- SLA compliance: binary reward for meeting latency target (weight: 0.2)
- Efficiency bonus: fraction of budget saved (weight: 0.1)

### 3.2 Simulator

Since real GPU benchmarking is expensive, we build a calibrated simulator using published performance data from cloud providers and model benchmarks. The simulator takes a (hardware, model, config) tuple and a customer profile, and outputs estimated quality, latency, and cost. Hardware throughput numbers are derived from published benchmarks (e.g., Nebius Token Factory specs, NVIDIA MLPerf results). Model quality scores are calibrated against public leaderboards (LMSYS Chatbot Arena, HumanEval, MMLU).

### 3.3 RL Algorithms

We implement and compare four approaches, mapping directly to course topics:

1. **Epsilon-greedy bandit** (baseline) - treats each valid stack as an independent arm. Ignores customer profile context. Maps to K-Armed Bandits (Course Week 2).

2. **Tabular Q-Learning** - discretizes the customer profile into buckets and learns a Q-table over sequential layer decisions. Maps to TD Learning and Q-Learning (Course Weeks 6-7).

3. **DQN (Deep Q-Network)** - uses a 3-layer MLP (128 hidden units) to approximate Q-values over the continuous customer profile space. Experience replay and target network for stability. Maps to Function Approximation (Course Week 8).

4. **PPO (Proximal Policy Optimization)** - policy gradient approach for comparison against value-based methods. Maps to Policy Gradient / Actor-Critic (Course Weeks 9-10).

### 3.4 Online Continuous Optimization

After the initial recommendation is deployed, a contextual bandit monitors real workload metrics (actual traffic patterns, observed latency, cost) and suggests adjustments weekly. This models the non-stationary aspect of the problem: customer workloads drift over time.

---

## 4. Team Member Responsibilities

| Member | Responsibility | Deliverables |
|--------|---------------|-------------|
| Anupama | Simulation and data engineering: build the hardware/model performance simulator, customer profile generator, compatibility matrix, and Gymnasium environment wrapper | simulator/, profiles.py, environment.py, 1000+ generated customer profiles for training |
| Jay | RL algorithm implementation: implement bandit, Q-learning, DQN, and PPO agents, training loops, hyperparameter tuning, and contextual bandit for online optimization | agents/, online/, trained models, training curves |
| Vishnu | Evaluation and integration: design and run experiments, ablation studies, generate Pareto plots and per-use-case heatmaps, build Streamlit demo dashboard, write final report and presentation | evaluation/, app.py, all figures, final report |

**Timeline:**

| Week | Anupama | Jay | Vishnu |
|------|---------|-----|--------|
| 1-2 | Build simulator, validate against real benchmarks | Set up RL framework, implement bandit baseline | Literature review, experiment design |
| 3-4 | Refine simulator based on RL feedback, add edge cases | Implement Q-learning, DQN, begin PPO | Begin running experiments on bandit/Q-learning |
| 5-6 | Support online optimization simulator | Complete PPO, implement contextual bandit | Full evaluation, ablations, Pareto plots |
| 7-8 | Code cleanup, documentation | Hyperparameter tuning, final training runs | Demo dashboard, final report, presentation |

---

## 5. Metrics for Validation and Verification

We evaluate on 1000 randomly generated customer profiles across 6 use case types.

**Primary metrics:**
- **Recommendation quality:** average simulated task accuracy of the recommended stack (target: > 0.85)
- **Budget compliance rate:** percentage of recommendations within the customer's budget (target: > 90%)
- **SLA compliance rate:** percentage of recommendations meeting latency SLA (target: > 90%)
- **Cost efficiency:** average percentage of budget saved while meeting quality and SLA targets

**Comparative evaluation:**
- RL agents vs random baseline, heuristic rule-based system, and each other
- Pareto frontier plots: quality vs cost for each method
- Per-use-case breakdown: does the agent learn different strategies for chatbots vs code generation vs batch processing?

**Ablation studies:**
- Remove hardware-awareness: how much does cost increase?
- Remove precision tuning: how much does quality drop?
- Remove scaling policy: how often is SLA violated during simulated peak traffic?
- Single-tier only vs multi-tier: what is the quality/cost tradeoff?

**Online adaptation evaluation:**
- Simulate workload drift over 4 weeks
- Measure how quickly the contextual bandit adapts vs a static recommendation

---

## 6. Framework and Tools

| Component | Tool | Why |
|-----------|------|-----|
| RL environment | Gymnasium (OpenAI Gym successor) | Standard RL environment interface |
| RL algorithms | Stable-Baselines3 | Production-quality DQN and PPO implementations |
| Bandit/Q-learning | Custom implementation (NumPy) | Simple enough to write from scratch, better for course credit |
| Simulation | NumPy + custom Python | Lookup tables calibrated from real benchmarks |
| Visualization | Matplotlib, Seaborn | Pareto plots, heatmaps, training curves |
| Demo | Streamlit | Interactive dashboard for demo |
| Language | Python 3.10+ | Standard for RL research |

**Existing implementations we build upon:**
- Stable-Baselines3 (Raffin et al., 2021) for DQN and PPO algorithm implementations
- Gymnasium API specification for environment design
- Hardware performance numbers from Nebius Token Factory documentation and NVIDIA MLPerf benchmarks

**All training runs on CPU.** The RL agent is a small MLP. No GPU compute needed for the RL training itself.

---

## References

1. Raffin, A. et al. "Stable-Baselines3: Reliable Reinforcement Learning Implementations." JMLR, 2021.
2. Sutton, R.S. & Barto, A.G. "Reinforcement Learning: An Introduction." 2nd ed., MIT Press, 2018.
3. Nebius. "Token Factory: Managed Inference for Open-Source Models." nebius.com/services/token-factory, 2026.
4. Ong, I. et al. "RouteLLM: Learning to Route LLMs with Preference Data." ICLR, 2025.
5. Schulman, J. et al. "Proximal Policy Optimization Algorithms." arXiv:1707.06347, 2017.
6. Mnih, V. et al. "Human-Level Control Through Deep Reinforcement Learning." Nature, 2015.
7. Chen, X. et al. "Lodestar: An Online-Learning LLM Inference Router." arXiv:2606.00946, 2026.
8. Zhang, Y. et al. "Temporal-Aware GPU Resource Allocation for Distributed LLM Inference via RL." arXiv:2507.10259, 2026.
