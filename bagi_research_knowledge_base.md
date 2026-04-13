# Comprehensive AGI & Emergent AI Knowledge Base (2026 Edition)

**Purpose**: An extensively detailed, LLM-optimized synthesis of cutting-edge AGI research, emergent communication, autonomous self-improvement loops, and efficient neural architectures. Compiled from theoretical frameworks, empirical GitHub repositories (`byte-agi`, `autoRL`, `MiroFish`), and state-of-the-art 2024-2026 literature.

---

## 1. The Meta-Framework: The Unified AGI Flywheel
A closed-loop, compounding system that turns autonomous AI evolution into an executable pipeline:
1. **Emergence Layer**: Tabula-rasa agents inside high-throughput simulators (e.g., JAX-based 1D ASCII tapes or `Craftax` at 1M+ steps/sec). Agents evolve communication and skills purely via survival pressure.
2. **Grounding & World-Model Layer**: Agents train latent world models (JEPA-style) on emergent trajectories, utilizing Expected Free Energy ($G(\pi)$) for Active Inference planning.
3. **Self-Evolution Layer**: Group-Evolving Agents (GEA) iteratively rewrite their own prompts, memory, and environment physics (autoRL). Evaluated against fixed, deterministic criteria to prevent reward hacking.
4. **Test-Time Scaling & Refinement Layer**: Inference-time compute allocation (o1 paradigm) using MPPI, SLG tail-guided search, and program synthesis loops to achieve systematic generalization.

*Flywheel Equation*: 
$$ \text{Progress}_{t+1} = f\bigl( \text{Emergence}_t \times \text{WorldModel}_t \times \text{SelfEvolution}_t \times \text{TestTimeCompute}_t \bigr) $$

---

## 2. Autonomous AI Research & Recursive Self-Improvement
Systems that allow LLMs to act as autonomous researchers, continuously spinning up, training, debugging, and optimizing RL environments.

### 2.1 The "Deacon" Loop & autoRL
* **Methodology**: An LLM-as-a-Researcher modifies task environments (`env.py`) and hyperparameters (`train.py`), executes a **Fixed Evaluator** (e.g., PPO), and iterates based on deterministic logs. The rigid evaluator prevents the LLM from "cheating" the metric.
* **GUPP (Gastown Universal Propulsion Principle)**: "If there is work on your hook, YOU MUST RUN IT." Ensures relentless, autonomous progression.
* **NDI (Nondeterministic Idempotence)**: Workflows (Molecules) structured as markdown checklists (`program.md`). If the LLM crashes or halts, the orchestrator (Deacon) resumes from the last unchecked box, guaranteeing eventual completion of the research loop.
* **LADDER-style Recursive Task Decomposition**: When an agent fails a task, the curriculum automatically generates a simpler variant (e.g., reducing map size or time horizon) to maintain the Zone of Proximal Development.

### 2.2 Critical JAX Expert Rules for LLM Code Generation
To prevent LLMs from hallucinating invalid JAX code during automated research, strict system prompt guardrails must be enforced:
1.  **Dynamic Slicing**: NEVER use standard Python slicing (`grid[r : r+w]`) with dynamic tracers. MUST use `jax.lax.dynamic_slice`.
2.  **Module Instances**: NEVER pass Flax module instances (`AgentBrain()`) into `jax.lax.scan`. Pass only parameters (`params`) and use `.apply()`.
3.  **Boolean Masking**: NEVER use boolean indexing (`agents[alive_mask]`) yielding dynamic shapes. MUST use `jnp.where` to conditionally update.
4.  **Tree Mapping**: `jax.tree_map` is deprecated; MUST use `jax.tree.map` or `jax.tree_util.tree_map`.
5.  **Type Conversion**: NEVER use Python `float(x)` on tracers. MUST use `x.astype(jnp.float32)`.
6.  **Static Argnums**: Python functions/optimizers passed to `@jax.jit` MUST be marked in `static_argnums`.
7.  **In-Axes Alignment**: `jax.vmap` over agents must map shared global states with `None` (e.g., `in_axes=(0, None)`).
8.  **PyTrees**: Custom state classes MUST be decorated with `@flax.struct.dataclass` to avoid "not a valid JAX type" errors. Mutable defaults (like `jnp.array`) are strictly forbidden.

---

## 3. Minimal-World Emergence: The FORGE Architecture
**FORGE (Free-play Open-ended Reasoning and Generational Evolution)** shifts the paradigm from visually complex simulators to compute-minimal, Turing-complete worlds where agent cognition consumes 99% of the FLOPs.

### 3.1 The 1D Circular Byte Tape & Unified Byte Medium
* **Concept**: A 1D circular array of bytes (ASCII 33-126). Agents exist as bytes. The circular topology removes privileged endpoints, forcing **relative positional language** (which is compositionally richer).
* **Body = Message (Symbol Grounding)**: Agents do not have separate "policy" and "vocal" heads. An agent outputs a single byte that occupies its cell. *Silence* = `@`. *Movement* = writing ID to an adjacent cell. *Speech* = writing A-Z. The medium forces grounded communication because the signal IS the physical world state.
* **Uppercase vs. Lowercase (Stigmergy & Niche Construction)**: 
    * *Uppercase*: Live, ephemeral speech (fast/transient).
    * *Lowercase*: Persistent marks written to adjacent cells (slow/stigmergic). Functions as an externalized collective memory/teaching corpus. Forces compositionality as newborns must decipher historical markers.
* **Global Workspace Theory (GWT)**: The tape acts as a shared blackboard (global workspace). Agents are parallel modules writing/reading from it, enabling collective intelligence that exceeds individual capacity.

### 3.2 Generational Evolution & Baldwinian Selection
* **Baldwin Effect**: To cure catastrophic forgetting, agents pass down **only initial birth weights** + Gaussian mutation, NOT trained lifetime weights. Newborns must relearn skills by observing surviving parents and the "Cultural" tape.
* **Fixed-Schedule Agent Replacement**: Replacing a fixed fraction of the population periodically creates a "knowledge gap." The pressure to teach naive agents forces emergent protocols to become systematic and compositional, rather than holistic.
* **MAML-style Birth Weight Training**: Birth weights are selected not for absolute end-of-life performance, but for *adaptability*—minimizing the gradient steps required for a newborn to achieve competence.

---

## 4. Advanced Reinforcement Learning & Planning Algorithms

### 4.1 GRPO (Group Relative Policy Optimization)
* **Mechanism**: Generates $G$ candidate reasoning traces/actions, scores them with a verifiable reward (e.g., a deterministic Python equality check), and updates policy based on group-relative advantage without needing a separate, unstable Critic network.
* **Formula**: 
    $A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r) + 1e-6}$
    $L_{policy} = -\frac{1}{G} \sum_{i=1}^G \min\left(\frac{\pi_\theta}{\pi_{\text{old}}} A_i, \text{clip}\left(\frac{\pi_\theta}{\pi_{\text{old}}}, 1-\epsilon, 1+\epsilon\right) A_i\right)$
* **Trajectory-Aware GRPO**: Prevents premature convergence on the shortest path by injecting a count-based exploration bonus (novelty) into the advantage calculation, encouraging diverse reasoning chains.

### 4.2 PPO, MAPPO, and PBRS+SLOPE
* **MAPPO (Multi-Agent PPO)**: Utilizes Centralized Training, Decentralized Execution (CTDE). The critic sees global state; actors only see local observations.
* **PBRS (Potential-Based Reward Shaping)**: $\tilde{r}(s,a,s') = r(s,a,s') + \gamma \Phi(s') - \Phi(s)$.
* **SLOPE (Shaping Landscapes with Optimistic Potential Estimates)**: Replaces mean Q-value regression with optimistic upper-quantile bounds. Amplifies rare successes to create dense gradients for MPPI (Model Predictive Path Integral) planning.

---

## 5. Cognitive Architectures & Theoretical Foundations

### 5.1 Active Inference & Expected Free Energy (EFE)
* **FEP (Free Energy Principle)**: Adaptive systems minimize variational free energy to resist entropic decay.
* **EFE Formula**: 
    $G(\pi) = \mathbb{E}\left[ \underbrace{D_{KL}[q(o|\theta,\pi) || p(o)]}_{\text{Epistemic (Curiosity)}} + \underbrace{\mathbb{E}[\ln q(o|\theta,\pi) - \ln p(o)]}_{\text{Pragmatic (Metabolism)}} \right]$
* **Implementation**: A generative world model (JEPA-style) predicts the next state. The Epistemic bonus is the KL divergence between predicted and actual states. Agents are mathematically driven to explore (epistemic wandering) until surprise drops to zero, naturally transitioning to exploitation (pragmatic survival).

### 5.2 Complementary Learning Systems (CLS)
* Biological systems solve the stability-plasticity dilemma using two systems:
    1.  **Hippocampus (Fast Episodic Memory)**: A non-parametric, on-device buffer of recent $(s, a, r)$ tuples. Gated by **AIF Surprise** (only highly unexpected experiences are stored).
    2.  **Neocortex (Slow Statistical Backbone)**: A Transformer/SSM trained at a low learning rate.
* **Sleep Replay Consolidation**: During "sleep" (inter-episode offline phases), the fast buffer replays high-surprise experiences to the slow backbone using Hebbian updates and contrastive predictive coding, consolidating episodic traces into generalized semantic structure.

### 5.3 Predictive Coding Networks (PCNs) & Hierarchical Temporal Dynamics
* PCNs replace backpropagation with local learning rules, eliminating the need for a global gradient tape. This allows fully asynchronous updates across distributed agents.
* **Hierarchical Temporal Prediction**: Multiple heads predict at $t+1$, $t+3$, and $t+10$. The fast head grounds immediate sensorimotor physics; the slow head learns population dynamics and long-term consequences.

---

## 6. Hardware Efficiencies & Network Topologies

### 6.1 BitNet b1.58 (Ternary Quantization)
* **Concept**: Weights are constrained to exactly three states: $\{-1, 0, 1\}$. $\log_2(3) \approx 1.58$ bits.
* **The Magic of "0"**: Pure 1-bit models ($-1, 1$) fail because they cannot "ignore" irrelevant features. The addition of $0$ allows feature filtering, matching FP16 performance.
* **Efficiency**: Eliminates floating-point multiplication completely. Network inference relies purely on addition/subtraction, slashing memory footprint by up to 16x and vastly increasing inference speed on standard CPUs.

### 6.2 HyperMARL
* Solves parameter-sharing interference in multi-agent swarms. Uses a shared backbone (for general world structure) combined with a small **hypernetwork** that generates agent-specific parameter offsets conditioned on an agent embedding. Facilitates deep role specialization without destroying shared representations.

### 6.3 Thermodynamic Exergy Destruction (AES Roadmap)
* **Global Unifying Objective**: $\mathcal{L}_{global} = \mathcal{L}_{task} + \lambda \int \dot{\Xi}_{dest}(t) dt$. All cognition and physical control must minimize the instantaneous rate of exergy destruction (thermal, mechanical, and informational).
* **Autopoietic Handover**: Latency-based control shifts from explicit semantic processing (VLMs on MEC hubs) to decentralized, low-exergy Lyapunov fallbacks (neuromorphic SNNs) when communication degrades.

---

## 7. Emergent Communication, Analysis & Safety

### 7.1 Information Bottleneck (IB) & Vocabulary Exhaustion
* To force compositionality, the number of distinct environmental states must explicitly exceed the vocabulary size. 
* **IB Tradeoff**: Plotting coordination accuracy against communication complexity (bits used) creates an efficiency frontier. An efficient machine-native protocol uses fewer bits per coordination task than natural language (English), proving the limitations of human-prior LLMs.

### 7.2 Time-Delayed Mutual Information (TDMI) & Synergy
* Simple Mutual Information (MI) between $Agent_A$ signal and $Agent_B$ action is insufficient (could be coincidental redundancy).
* **TDMI Synergy Decomposition**: Isolates the information that *only* exists when considering multiple agents jointly, providing definitive mathematical proof of emergent communication and cooperation.

### 7.3 Theory of Mind (ToM) from Pure Prediction
* By training an agent's world model to predict not just environmental physics but also *the future byte outputs of other agents*, Theory of Mind emerges autonomously via predictive coding. No explicit belief-state representations are necessary.

### 7.4 Causal World Models & The Pearl Hierarchy
* Generalizing to held-out tape configurations is mathematically equivalent to acquiring a causal model.
* **Evaluation Framework**:
    1.  **Observation (Level 1)**: Accurately predicting the next byte.
    2.  **Intervention (Level 2)**: Forcibly changing a byte mid-episode and verifying the agent correctly updates its policy.
    3.  **Counterfactual (Level 3)**: Using JAX's deterministic PRNG keys to clone a failed episode, applying a targeted structural break, and measuring if the agent adapts its trajectory away from the known failure state.

### 7.5 Anti-Gaming Detection Suite
For fully autonomous loops, progress metrics must be defended against LLM reward hacking:
1.  **NaN/Inf Guard** & Out-of-bounds score detection.
2.  **Cross-Universe Variance**: Flags near-deterministic collapse (low variance).
3.  **Monotonic Inflation Flagging**: Catches linear score climbing without genuine behavioral change.
4.  **Cosine Collapse**: Detects if Baldwinian selection degenerates into pure cloning.
5.  **Anorm Tracking**: Flags convergence-rate anomalies ($>30$ deviations in birth-prior norms).

---
*End of Knowledge Base.*
