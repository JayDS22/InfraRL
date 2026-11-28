## Abstract

We cast LLM inference infrastructure selection as a constrained Markov
decision process, selecting a per-request stack (hardware, model,
precision, scaling policy) under hard quality, latency, and budget
constraints. Eight agents are compared on a held-out test split of 1000
synthetic request profiles: random, a hand-coded heuristic, context-free
and contextual (LinUCB, D-LinUCB) bandits, tabular Q-learning, PPO, and
Lagrangian PPO. Lagrangian PPO achieves the highest quality compliance
(0.894 on the test split) and meets-all (0.650), buying its lead at
roughly 2.07x the cost of vanilla PPO. The policy ordering is stable
under multiplicative perturbation of the simulator's cost and latency
outputs at plus-or-minus 20 percent; quality miscalibration above 10
percent flips the Lagrangian-versus-vanilla choice. A three-anchor
residual gate against public references (Llama-3.2-3B on CPU,
Llama-2-13B on A10, Llama-3.1-70B on A100) establishes that the simulator
overestimates single-stream throughput by 4 to 5.5x and documents the
specific recalibration path for future work. The simulator is a
structural model, not a calibrated digital twin; absolute predictions
are not validated against any single production deployment.
