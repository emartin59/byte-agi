# Advanced AGI & Autonomous AI Research Knowledge Base (2026 Synthesis)

**Purpose**: An extensive, LLM-optimized synthesis of the latest AGI research, focusing on recursive self-improvement, emergent multi-agent communication, and advanced cognitive architectures. Compiled from state-of-the-art surveys, GitHub repositories (`byte-agi`), and 2024-2026 architectural breakthroughs.

---

## 1. Recursive Self-Improvement & Self-Evolving Agents
Systems that autonomously acquire experience, refine their logic, and update their own architectures (prompts, tools, memory, or parameters) without human intervention.

### 1.1 The Self-Evolution Framework
* **Experience Acquisition**: Agents gather data through *Trajectory* (environmental interaction), *Direct* (observing expert datasets), or *Reflection* (internal critique and simulation).
* **Refinement Loop**: Driven by a Proposer-Verifier architecture. The Proposer LLM generates hypotheses or code updates (e.g., modifying RL environment physics), and the Verifier tests them against deterministic criteria (to prevent reward hacking).
* **Evolution Targets**: 
  * *Prompt Evolution*: Meta-prompting to refine instructions.
  * *Memory Evolution*: Abstracting episodic memory into semantic rules.
  * *Tool Evolution*: Synthesizing new executable functions (e.g., Python tools) for reuse.

### 1.2 Autonomous Research Loops (The "Deacon" Pattern)
* **LLM-as-a-Researcher**: A structured loop where the LLM writes JAX/Python code, executes it in a sandbox, and parses error logs or score outputs. 
* **Nondeterministic Idempotence (NDI)**: Research tasks are tracked via markdown checklists. If the agent crashes or is rate-limited, the orchestrator resumes from the exact point of failure, ensuring relentless progression.

---

## 2. Multi-Agent Emergence & Tabula Rasa Worlds
Moving beyond LLM text-prediction by forcing agents to develop grounded intelligence through survival pressure in compute-minimal environments.

### 2.1 Compute-Minimal Substrates (1D ASCII Tapes)
* **Concept**: Turing-complete 1D or 2D worlds rendered as byte arrays. Agents and objects exist as ASCII characters.
* **Unified Byte Medium (Stigmergy)**: Agents do not possess separate "communication channels." Their physical body *is* the message. 
  * *Uppercase (Speech)*: Ephemeral broadcast of intent.
  * *Lowercase (Writing)*: Persistent marks left on the tape. Functions as an externalized collective memory (stigmergy), forcing compositionality as new generations must decipher historical markers.

### 2.2 Baldwinian Evolution & Cultural Transmission
* **Mechanism**: To cure catastrophic forgetting, populations are periodically culled. Newborns inherit **only initial birth weights** (not lifetime-trained weights) + Gaussian mutation.
* **Knowledge Gap**: Replacing a fraction of the population creates a learning gap. Newborns must re-learn coordination by observing surviving parents, driving the emergent language to become highly systematic and compositional.

---

## 3. Core Algorithms & Training Paradigms

### 3.1 Group Relative Policy Optimization (GRPO)
* **Mechanism**: Generates multiple candidate trajectories/reasoning chains ($G$), scores them using a verifiable deterministic reward (e.g., math or code correctness), and computes a group-relative advantage. Eliminates the need for a separate, unstable Critic network.
* **Formula**: 
  $$A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r) + 1e-6}$$
* **Trajectory-Awareness**: To prevent premature convergence on the shortest path, a novelty/epistemic bonus is added to the advantage calculation, promoting exploration of diverse reasoning chains.

### 3.2 Active Inference & Expected Free Energy (EFE)
* **Framework**: Agents minimize variational free energy. Action selection is driven by Expected Free Energy.
* **Formula**: 
  $$G(\pi) = \mathbb{E}\left[ \underbrace{D_{KL}[q(o|\theta,\pi) || p(o)]}_{\text{Epistemic (Curiosity)}} + \underbrace{\mathbb{E}[\ln q(o|\theta,\pi) - \ln p(o)]}_{\text{Pragmatic (Metabolism)}} \right]$$
* **Implementation**: A world model predicts the next state. The Epistemic bonus (surprise) drives the agent to explore novel regions until the environment is mapped, at which point it transitions to Pragmatic survival.

---

## 4. Cognitive Architectures & World Models

### 4.1 Complementary Learning Systems (CLS)
* **Hippocampus (Fast Episodic Memory)**: An on-device buffer of recent transitions. Gated by AIF Surprise—only highly unexpected events are stored.
* **Neocortex (Slow Statistical Backbone)**: A Transformer or Mamba SSM trained at a low learning rate.
* **Sleep Replay Consolidation**: During offline phases, the fast buffer replays high-surprise experiences to the slow backbone, converting episodic instances into general semantic rules.

### 4.2 Predictive Coding Networks (PCNs)
* Replaces global backpropagation with local learning rules. Enables fully asynchronous updates across distributed swarms. Inference depth is dynamic: surprising inputs trigger deeper iterative refinement steps.

### 4.3 Test-Time Compute (System-2 Reasoning)
* **o1/o3 Paradigm**: Scaling intelligence at inference time rather than training time.
* Uses Monte Carlo Tree Search (MCTS) or Model Predictive Path Integral (MPPI) to explore multiple reasoning chains, guided by Process Reward Models (PRMs) that score intermediate logical steps.

---

## 5. Hardware Efficiencies & Safety Protocols

### 5.1 BitNet b1.58 & Ternary Hardware
* **Mechanism**: Neural network weights are constrained to $\{-1, 0, 1\}$. 
* **Impact**: Eliminates floating-point multiplication, relying purely on addition/subtraction. Slashes memory requirements and drastically increases tokens-per-second, allowing massive multi-agent simulations on minimal edge hardware.

### 5.2 JAX/TPU Optimization
* **Vectorization**: Uses `jax.vmap` and `jax.lax.scan` to run thousands of parallel environments on a single GPU/TPU.
* **HyperMARL**: Solves parameter-sharing interference in swarms by using a shared transformer backbone + tiny agent-specific hypernetworks that generate personalized parameter offsets.

### 5.3 Anti-Gaming & Evaluation
* **Pearl Causal Hierarchy**: Evaluates agent world models via Observation (next-byte prediction), Intervention (mid-episode state manipulation), and Counterfactuals (cloned trajectory divergence).
* **Anti-Gaming Suite**: Automated orchestrators track metric validity via NaN guards, Cross-Universe Variance (detecting deterministic collapse), and Cosine Collapse (detecting genetic cloning masquerading as learning).

---
*End of Synthesis.*
