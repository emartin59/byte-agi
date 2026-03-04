# byte-agi

I had help from generative AI (Gemini and xAI) to come up with this text and the project concepts and code. Almost all text/code is AI generated. The hope for the project is to create all of the below phases and test them, and for the earlier phases to be runnable for free within a single Kaggle notebook with the GPU P100 selected on the right-side menu. Not there yet.

byte-agi is a minimal, biologically-plausible, mathematically-grounded sandbox for emergent intelligence.

Everything is a single byte (0–255). Vision is blind to self. Speech is visual (your body literally changes glyph). The only objective is Expected Free Energy (FEP/EFE). The code stays in pure JAX/Flax so every line compiles to GPU/TPU and scales with `vmap`/`pmap`. No Unity, no separate audio stream, no hand-crafted rewards or exploration bonuses—only physics, metabolism, curiosity, and death.

The roadmap is deliberately staged: **never change the learning algorithm and the environment at the same time**. Each phase has clear success metrics, required libraries, and expected emergent behaviors.

This is a clean launchpad that lets language, cooperation, cultural transmission, and open-ended tool use emerge.

### Phase 1: Hello World – Berry Hunter (Single Agent, Trainable)
* **Environment:** 16-byte 1D tape, one `@` agent, one Berry.
* **Vision:** Blind-self vision using a 9-tile window (4 left + 4 right), with the center masked to EMPTY.
* **Proprioception:** Concatenates current hunger + last action + last vocal (even if Phase 1 vocal=0).
* **Brain:** `nn.Embed(256, hidden_dim)` → LSTMCell → policy (3 actions) + predicted next 9-tile vision.
* **Training:** Full episode rollouts via `jax.lax.scan` (50–100 steps) so gradients flow end-to-end.
* **Action Sampling:** Categorical + small entropy bonus.
* **Metabolism:** Real hunger decay + regeneration (agent must keep eating).
* **Objective:** Simple extrinsic Free Energy (hunger error) trained with real gradients.
* **Success Metric:** Agent reliably finds and eats the berry; Free Energy visibly → 0.
* *Debug Output: Tick | Pos | Hunger | Total_FE + live tape render.*
### Phase 2: Stag Hunt – Visual Language Birth & Vectorization
* **Expanded Environment:** Expand to 32-byte tape. Add Button, Door, Core.
* **Communication:** Agents now output vocal (0=silent `@` or 1–26=A–Z). Speaking changes their body byte and applies 0.01 metabolic tax.
* **Sensory Input:** Same blind 9-tile vision + proprioception (which now includes own energy).
* **Brain:** policy_logits (3) + vocal_logits (27) + predicted_vision head.
* **Vectorization (`jax.vmap`):** Vectorize over 1,024 parallel universes (8 agents each) before implementing Phase 3. This forces the use of `jnp.where` instead of Python `if/else` statements, ensuring the framework scales cleanly without requiring a massive code rewrite when complex epistemic math is added later.
* **Deployment:** Scripted demo first, then train both agents jointly.
* **Success Metric:** One agent learns to step on the button and flash a consistent capital letter; the other sees it in its local view and walks through the opened door to the Core.
* *Debug Output: Rendered tape shows exactly the “X opens door” moment.*
### Phase 3: Deep Active Inference – Curiosity & Imagination
* **Architecture:** Split brain into Encoder → Transition Model (variational) → Policy.
* **Expected Free Energy (EFE) Loss - Extrinsic:** Hunger/energy error.
* **Expected Free Energy (EFE) Loss - Epistemic:** $\text{KL}(\text{predicted next vision} \parallel \text{actual next vision})$ after one-hot encoding the 9 tiles.
* **Expected Free Energy (EFE) Loss - Total:** Total_FE = extrinsic + β·epistemic (β anneals from high to low).
* **Transition Model:** Trained with cross-entropy predictive coding (the agent literally learns to “imagine” the next tape state).
* **Exploration:** No hand-coded exploration—the math forces pure epistemic wandering until the map is known.
* **Success Metric:** Beautiful loss curves where epistemic loss is high at spawn, drops to zero, and pragmatic takes over. Agents explore unknown tiles then exploit.
### Phase 4: Emergent Physics – Cellular Automata Replace All Hard-Coding
* **Physics Engine:** Replace every button/door/core rule with local Rule-110-style cellular-automaton updates (only 3–5 bytes interact locally).
* **Gameplay:** Agents must discover crafting, doors, and resources through experiment.
* **Constraints:** Keep metabolic tax and blind vision.
* **Language Analysis Suite:** Automatically logs vocal bigram/trigram frequencies, mutual information (vocal ↔ button state), and convention success rate.
* **Success Metric:** Pareto distribution in vocabulary (structured language emerges, not random flashing). Agents invent new physics uses.
### Phase 5: Evolution & Cultural Transmission – Generational Turnover
* **Life Cycle & Lifetime Fitness:** Every 500 steps, cull the population. Crucially, survival is judged on an agent's "Lifetime Fitness" (an exponential moving average of their hunger across dozens of episodes) rather than their performance on a single map. This eliminates the "Luck vs. Skill" problem of procedural generation, ensuring we cull agents that are genuinely unfit rather than those who just rolled a difficult map.
* **Reproduction:** Birth new agents by copying survivor weights + small Gaussian mutation (or zero-init for pure cultural reset).
* **Lamarckian Evolutionary Mechanics:** Because the agents learn via gradient descent during their lifetime, this simulates the Baldwin Effect. Newborns inherit the visual 26-cap language from parents via observation only—no weight transfer required for culture to spread.
* **Unsupervised Environment Design (UED) Level 1:** Randomize the map layout every episode (shifting the Button position, altering the Wire length, moving the Door and Core). This mathematically prevents agents from over-fitting to a specific trajectory, forcing their Imagination (Transition Model) to truly understand the physics of the world regardless of configuration.
* **Scaling:** Continue `vmap` scaling and EFE training across generations.
* **Success Metric:** The “X = open door” convention (or better ones) survives and refines across generations and across shifting map layouts. You literally watch cultural evolution in the vocab logs as they adapt to procedural generation, proving the agents are passing down genuine, generalized problem-solving skills.
### Phase 6: The AGI Runway – Hyperscale & External Translation
* **Distributed Training:** `jax.pmap` across multi-GPU/TPU cluster (same code, instant scaling).
* **Efficiency:** Integrate L-Mul (Linear-complexity Multiplication from BitEnergy AI) to replace FP multiplies in the LSTM core with integer adds, enabling billions of parameters at extreme efficiency.
* **Advanced UED & Open-Endedness:** Expand procedural generation to create complex, multi-step 1D puzzles (e.g., multiple doors, decoy buttons, logic gates formed by crossing wires, randomly scattered and regenerating cores). Because the Lifetime Fitness metric from Phase 5 safely evaluates agents across varying difficulties, the 1D tape can now seamlessly become a continuous, unpredictable, and infinite survival landscape without the risk of accidentally culling your smartest agents due to bad RNG.
* **The 1D Constraint:** Keep the environment strictly 1D until everything above works perfectly. 2D pathfinding consumes 90% of a neural network's capacity in standard RL. Restricting it to 1D ensures 100% of the compute is spent on language, logic, and theory of mind. Optional 2D upgrade (16×16 grid with 7×7 local view) serves as the ultimate final test of generalization.
* **“God” LLM Translator:** Every 10 steps feed the last 32 bytes + vocal history to an external LLM and ask it to translate the emerging 26-cap language into English. Purely for human insight.
* **Open-Source License:** MIT + "research use encouraged" so independents / labs / academia can freely play.
## AGI Roadmap (2026–2029)
### Phase 7: Actor-Critic Foundations – Strict 1D Expansion
**Timeline:** March–April 2026 (3–5 weeks)  
**Compute:** Single Kaggle TPU v5e-8 (exactly as Phase 6)  
**Goal:** Fix credit assignment so agents can plan over 200–300 steps. Prove GAE + Value Head works before touching 2D. Incorporate explicit Expected Free Energy (EFE) for natural curiosity and planning.  
**Key upgrades**  
- Brain: Actor-Critic with Policy head, Vocal head, and new Value head (critic predicts remaining hunger/surprise).  
- Learning: Full Generalized Advantage Estimation (GAE λ=0.95, γ=0.99) + value loss, blended with EFE (EFE = pragmatic [goal surprise via hunger minimization] + epistemic [information gain via transition CE loss]). Use pymdp's JAX backend for EFE computation, where EFE replaces or augments pure rewards in advantages.  
- Early termination: Episode ends the exact tick any agent eats a core. Terminal bonus = 1.0 – mean final hunger (forces speed and coordination).  
- World: 1D tape expanded to **256 tiles**, **6 agents** per universe (stacking allowed — agents can occupy same tile to prevent traffic jams).  
- Extreme UED: 0–3 decoy buttons, wire lengths 20–200 tiles, randomized real-button logic, occasional double-core vaults.  
- Brain addition: Cheap sliding-window attention over last 32 steps on top of LSTM (HIDDEN_DIM=128).  
- Diagnostics added: success rate %, average episode length, role entropy, vocal mutual-information tables, bigram heatmaps, broadcasting frequency, EFE breakdown (pragmatic vs. epistemic components).  
**Success criteria**  
- Success rate >85 % on 300-step tasks.  
- Average episode length drops from ~250 → <80 ticks.  
- Emergent language expands (roles: scouts, pressers, confirmers; broadcasting signals; “names” for agents).  
- Stable training (no gradient explosion).  
- EFE drives better decoy handling and faster core-eating (2–5x efficiency in exploration, as per Intermittent AIF benchmarks).  
**Dependencies:** `pip install pymdp` (JAX backend for EFE implementation).  
### Phase 8: First 2D World & Memory Upgrade
**Timeline:** May–August 2026 (3–4 months)  
**Compute:** Same TPU v5e-8 (still fits 1024 envs).  
**Goal:** Introduce spatial complexity while enhancing memory and navigation with AIF elements for robust exploration in uncertain environments.  
**Key upgrades**  
- World: 32×32 → 64×64 grid using **CAX** library (ICLR 2025 oral — 10–100× faster 2D CA on TPU).  
- Physics: Wireworld-style cellular automata (conductors, sources, transistors/gates, structural blocks).  
- Agents: 8–16 per universe, 9×9 local vision, 5-tile vocal radius.  
- Actions: Mine + Place (limited to conductors and basic gates). Core vaults still provide reliable reward.  
- Brain: Replace LSTM with small Mamba or sliding-window Transformer memory (CAX examples available).  
- AIF Integration: Add AIF navigation system (adapted from de Tinguy et al., IWAI 2025) for real-time pathfinding in ROS2-compatible style, but implemented in pure JAX/CAX. Agents predict 3–5 steps ahead via forward generative models, minimizing surprise in partially observable grids.  
- Fitness: Total energy harvested by the tribe + bonus for machine complexity (gate count).  
- Cultural transmission: Top 25 % agents’ memory embeddings seeded into newborns.  
**Success criteria**  
- Agents build simple circuits to reach cores.  
- Clear division of labor (scouts vs builders vs signal-relay specialists).  
- Reusable “blueprints” communicated via glyphs across the map.  
- Success rate >80 % on 500-step tasks.  
- AIF navigation boosts epistemic drive, leading to emergent signal towers and better handling of occluded vaults.  
**Dependencies:** `pip install cax` (already TPU-optimized); adapt de Tinguy et al. ROS2 code to JAX (fork their arXiv repo for navigation module).  
### Phase 9: Turing Sandbox – Open-Ended 2D
**Timeline:** September 2026–March 2027 (6–7 months)  
**Compute:** TPU v5e-8 or small pod slice.  
**Goal:** Achieve open-ended evolution with hierarchical multi-agent AIF for complex machine-building and strategic interactions.  
**Key upgrades**  
- World: Full **256×256** CAX grid with procedural biomes and scattered resource vaults (no pre-built solutions).  
- Agents: 16–32 per universe.  
- Physics: Fully Turing-complete blocks (transistors, logic gates, etc.).  
- LLM-in-the-loop: Your “God” translator runs on top 10 % every 250 generations → extracts dictionary → injects as auxiliary loss or embedding prior.  
- AIF Integration: Full hierarchical multi-agent AIF (VERSES-style from "Mobile Manipulation" arXiv:2507.17338: high-level discrete skills for roles like scouts/engineers + low-level continuous control). Factorized generative models (from Jaime et al., AAMAS 2025) allow private beliefs about other agents for strategic planning. Use Variational Bayes Gaussian Splatting for online 3D-like mapping in the 2D grid.  
- Fitness: Purely lifetime tribe energy harvested + complexity of persistent machines left on the map.  
- Open-endedness: Episodes run until starvation or 2,000 steps; new vaults spawn dynamically when mastery is detected.  
**Success criteria**  
- Emergent hierarchy, named agents, physical signal towers, and cultural traditions that survive generational turnover.  
- Agents invent and reuse complex machines across generations.  
- Vocabulary evolves into proto-grammar (conditional signals, planning statements).  
- Hierarchical AIF achieves >60% success on rearrangement-like tasks (e.g., multi-vault coordination), outperforming pure RL baselines without pre-training.  
**Dependencies:** Fork VERSES AI "Mobile Manipulation" repo (arXiv:2507.17338) for hierarchical AIF; integrate Jaime et al. factorized models (AAMAS 2025 code).  
### Phase 10: Embodiment in High-Fidelity Simulation
**Timeline:** April–December 2027 (8–9 months)  
**Compute:** TPU pod + GPU cluster for sim rendering.  
**Goal:** Bridge to embodied robotics with AIF controllers for whole-body coordination and zero-shot adaptation.  
**Key upgrades**  
- Interface: Byte-grid vision (9×9 or 11×11) to JAX-native robotics simulators (MuJoCo, Brax, or Isaac Gym via JAX).  
- New modality: Audio (spoken commands become byte streams on the grid).  
- Tool-use: Special glyphs let agents call external JAX functions (math, search, code execution).  
- AIF Integration: VERSES/Fujii hierarchical controllers (from IEEE 2025 paper) for stable whole-body control, minimizing surprise in pixel-based or byte-grid inputs. Use temporally hierarchical world models for long-horizon planning in sparse-reward sim environments. Training stays fully in simulation (no physical hardware yet).  
**Success criteria**  
- Zero-shot sim-to-sim transfer across different robot morphologies.  
- Agents use tools and language to solve embodied tasks (navigation, manipulation, collaborative construction).  
- AIF controllers outperform RL in adaptation to noise/dynamics mismatch, with stable control in tasks like reaching or multi-agent rearrangement.  
**Dependencies:** `pip install mujoco brax isaacgym` (JAX wrappers); fork Fujii et al. repo for hierarchical world models.  
### Phase 11–12: Recursive Self-Improvement & Multi-Agent Societies
**Timeline:** 2028 (12–18 months total)  
**Compute:** Full TPU pod / Pathways cluster.  
**Goal:** Enable self-improvement in large-scale societies with AIF for decentralized coordination and optimization.  
**Key upgrades**  
- Meta-evolution: Agents propose and test small architecture mutations (NAS on Mamba/Transformer core).  
- Scale: 1,000–10,000+ agents across multiple universes with specialization and “trade”.  
- Environments: Indefinite runtime; new puzzles and biomes spawn automatically on mastery.  
- AIF Integration: Orchestrator-style monitoring (from Beckenbauer et al., OpenReview 2025) with attention-inspired tracking of agent-environment interactions for global performance optimization. Add agentic rulebooks (from Constant et al., Frontiers 2025) for sustainable, decentralized multi-agent coordination in environmental tasks.  
- Full cultural evolution metrics (dictionary stability, tradition inheritance).  
**Success criteria**  
- Agents improve their own training loop or memory architecture.  
- Stable multi-agent societies with division of labor, markets, and long-term planning.  
- AIF enables efficient decentralized optimization, with societies adapting to dynamic biomes without central control.  
**Dependencies:** Fork Beckenbauer et al. Orchestrator repo; integrate Constant et al. rulebooks for sustainability metrics.  
### Phase 13+: True AGI Deployment
**Timeline:** 2029 onward  
**Goal:** Deploy persistent, safe AGI societies with AIF priors for global stability and interpretability.  
**Key features**  
- Persistent multi-agent societies running 24/7 in simulation and real Spatial Web environments.  
- Grounded safety: Every action must reduce global free energy (active-inference prior).  
- LLM oracle access: Frontier models queried only through byte-level grounded channels.  
- AIF Integration: AIF safety priors (minimize global surprise) + Beckenbauer's multi-agent reflection for ongoing optimization and self-revision in decentralized systems.  
- Global audit log: Your LLM translator becomes the permanent interpretability layer.  
**Success criteria**  
- Societies maintain stability in open-world deployments, adapting to real-time changes while minimizing surprise.  
- Safe, interpretable behaviors emerge from AIF priors, with auditable generative models for all decisions.  
**Dependencies:** Integrate with Spatial Web standards (IEEE 2874) for real-world deployment; use VERSES AXIOM for unified perception/planning/control in production.
