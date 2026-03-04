# byte-agi

I had help from generative AI (Gemini and xAI) to come up with this text and the project concepts and code. Almost all text/code is AI generated. The hope for the project is to create all of the below phases and test them, and for the earlier phases to be runnable for free within a single Kaggle notebook with the GPU P100 selected on the right-side menu. Not there yet.

byte-agi is a minimal, biologically-plausible, mathematically-grounded sandbox for emergent intelligence.
Everything is a single byte (0–255). Vision is blind to self. Speech is visual (your body literally changes glyph). The only objective is Expected Free Energy (FEP/EFE). The code stays in pure JAX/Flax so every line compiles to GPU/TPU and scales with `vmap`/`pmap`. No Unity, no separate audio stream, no hand-crafted rewards or exploration bonuses—only physics, metabolism, curiosity, and death.

The roadmap is deliberately staged: **never change the learning algorithm and the environment at the same time**. Each phase has clear success metrics, required libraries, and expected emergent behaviors.
This is a clean launchpad that lets language, cooperation, cultural transmission, and open-ended tool use emerge.

phase-1-7.py and phase-1-6-sample-output.txt in the Files are from a prior roadmap/README.md. Significant tweaks from the prior roadmap start at Phase 4 below where CAX is now added at that stage.

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

### Phase 2: Deep Active Inference – Curiosity & Imagination (Single Agent)
* **Architecture:** Split brain into Encoder → Transition Model (variational) → Policy.
* **Expected Free Energy (EFE) Loss - Extrinsic:** Hunger/energy error.
* **Expected Free Energy (EFE) Loss - Epistemic:** $\text{KL}(\text{predicted next vision} \parallel \text{actual next vision})$ after one-hot encoding the 9 tiles.
* **Expected Free Energy (EFE) Loss - Total:** Total_FE = extrinsic + β·epistemic (β anneals from high to low).
* **Transition Model:** Trained with cross-entropy predictive coding (the agent literally learns to “imagine” the next tape state).
* **Exploration:** No hand-coded exploration—the math forces pure epistemic wandering until the map is known.
* **Success Metric:** Beautiful loss curves where epistemic loss is high at spawn, drops to zero, and pragmatic takes over. Agents explore unknown tiles then exploit.

### Phase 3: Stag Hunt – Visual Language Birth & Vectorization (Multi-Agent)
* **Expanded Environment:** Expand to 32-byte tape. Add Button, Door, Core.
* **Communication:** Agents now output vocal (0=silent `@` or 1–26=A–Z). Speaking changes their body byte and applies 0.01 metabolic tax.
* **Sensory Input:** Same blind 9-tile vision + proprioception (which now includes own energy).
* **Brain:** policy_logits (3) + vocal_logits (27) + predicted_vision head. (Agents already have intrinsic curiosity from Phase 2, so communication emerges naturally to reduce surprise about the other agent.)
* **Vectorization (`jax.vmap`):** Vectorize over 1,024 parallel universes (8 agents each) before implementing Phase 4.
* **Deployment:** Scripted demo first, then train both agents jointly.
* **Success Metric:** One agent learns to step on the button and flash a consistent capital letter; the other sees it in its local view and walks through the opened door to the Core.
* *Debug Output: Rendered tape shows exactly the “X opens door” moment.*

### Phase 4: Emergent Physics – Cellular Automata Replace All Hard-Coding
* **Physics Engine:** Replace every button/door/core rule with local Rule-110-style cellular-automaton updates (only 3–5 bytes interact locally). **All CA from this phase onward is implemented via the CAX library using fully differentiable Neural Cellular Automata (soft/continuous approximations with Gumbel-Softmax or Straight-Through Estimators).**
* **New Action:** Explicit “Interact/Modify” action head (4th action). The agent outputs a small localized perturbation vector that directly influences the CA cells immediately in front of it — this is how agents discover and perform crafting.
* **Gradient Management:** Use `jax.remat` (gradient checkpointing) on long rollouts and large grids to prevent VRAM explosion during the backward pass.
* **Monitoring:** Track divergence between the brain’s predicted next state and the actual CA update (STE instability can cause the model to drift from true physics).
* **Gameplay:** Agents must discover crafting, doors, and resources through experiment.
* **Constraints:** Keep metabolic tax and blind vision.
* **Language Analysis Suite:** Automatically logs vocal bigram/trigram frequencies, mutual information (vocal ↔ button state), and convention success rate.
* **Success Metric:** Pareto distribution in vocabulary (structured language emerges, not random flashing). Agents invent new physics uses.
* **Dependencies:** `pip install cax` (already TPU-optimized and fully differentiable).

### Phase 5: Evolution & Cultural Transmission – Generational Turnover
* **Life Cycle & Lifetime Fitness:** Every 500 steps, cull the population. Survival is judged on an agent's "Lifetime Fitness" (exponential moving average of hunger across dozens of episodes).
* **Reproduction:** Birth new agents by copying **initial birth weights only** + small Gaussian mutation. (Darwinian + Baldwin Effect — no direct copying of trained lifetime weights.)
* **Cultural Transmission:** Newborns must re-learn language and skills entirely through observation of surviving parents **and** the Cultural Elder (see below). This is true Baldwinian evolution and prevents catastrophic forgetting across generations.
* **Cultural Elder Seeding:** A simple “Elder” agent (lightweight behavioral clone or small LLM prompt) is occasionally present at birth. Newborns observe the Elder’s successful demonstrations alongside their parents right from the first generation.
* **Unsupervised Environment Design (UED) Level 1:** Randomize map layout every episode (button position, wire length, door/core placement).
* **Scaling:** Continue `vmap` scaling and EFE training across generations.
* **Success Metric:** The “X = open door” convention (or better ones) survives and refines across generations. You literally watch cultural evolution in the vocab logs.

### Phase 6: Actor-Critic Foundations – Strict 1D Polish
**Timeline:** March–April 2026 (3–5 weeks)  
**Compute:** Single Kaggle TPU v5e-8  
**Goal:** Fix credit assignment so agents can plan over 200–300 steps. Prove GAE + Value Head + explicit EFE works on 1D before any hyperscaling.  
**Key upgrades**  
- Brain: Actor-Critic with Policy head, Vocal head, and new Value head (critic predicts remaining hunger/surprise).  
- Learning: Full Generalized Advantage Estimation (GAE λ=0.95, γ=0.99) + value loss, blended with EFE (EFE = pragmatic goal surprise + epistemic information gain via transition CE loss).  
- Early termination: Episode ends the exact tick any agent eats a core. Terminal bonus = 1.0 – mean final hunger (forces speed and coordination).  
- World: 1D tape expanded to **256 tiles**, **6 agents** per universe (stacking allowed — agents can occupy same tile to prevent traffic jams).  
- Extreme UED: 0–3 decoy buttons, wire lengths 20–200 tiles, randomized real-button logic, occasional double-core vaults.  
- Brain addition: Cheap sliding-window attention over last 32 steps on top of LSTM (HIDDEN_DIM=128).  
- Diagnostics added: success rate %, average episode length, role entropy, vocal mutual-information tables, bigram heatmaps, broadcasting frequency, EFE breakdown (pragmatic vs. epistemic).  
**Success criteria**  
- Success rate >85 % on 300-step tasks.  
- Average episode length drops from ~250 → <80 ticks.  
- Emergent language expands (roles: scouts, pressers, confirmers; broadcasting signals; “names” for agents).  
- Stable training (no gradient explosion).  
- EFE drives better decoy handling and faster core-eating (2–5x efficiency in exploration).  
**Dependencies:** EFE computation must remain fully JAX-traceable (pure JAX KL + expected surprise). If pymdp breaks `jax.jit` inside the `lax.scan` loop, hand-roll the EFE math in pure JAX.

### Phase 7: The AGI Runway – Hyperscale & External Translation
**Timeline:** April–May 2026 (3–4 weeks)  
**Compute:** Multi-GPU/TPU cluster (pmap scaling)  
**Goal:** Take the now-stable Actor-Critic brain and scale it to hyperscale while adding the LLM translator.  
**Key upgrades**  
- Distributed Training: `jax.pmap` across multi-GPU/TPU cluster (same code, instant scaling).  
- Efficiency: Introduce **small Mamba** (State Space Model) memory and integrate L-Mul (Linear-complexity Multiplication from BitEnergy AI) to replace FP multiplies in the Mamba core with integer adds, enabling billions of parameters at extreme efficiency.  
- Advanced UED & Open-Endedness: Expand procedural generation to create complex, multi-step 1D puzzles (multiple doors, decoy buttons, logic gates formed by crossing wires, randomly scattered and regenerating cores).  
- “God” LLM Translator: Every 10 steps feed the last 32 bytes + vocal history to an external LLM and ask it to translate the emerging 26-cap language into English. Purely for human insight.  
- The 1D Constraint: Keep the environment strictly 1D until everything above works perfectly.  
**Success criteria**  
- Training runs stably at 1024+ parallel universes on TPU cluster.  
- Lifetime Fitness improves across generations.  
- LLM translator produces readable English interpretations of agent language.  
**Dependencies:** L-Mul implementation; external LLM API for translator.

## AGI Roadmap (Aspirational 2026–2030+)
**Note:** Phases 1–7 are grounded and achievable on a single Kaggle TPU v5e-8 or small cluster today. Phases 8–10 are ambitious but within reach for a dedicated solo/small-team developer. Phases 11–13 represent frontier research horizons that may take longer than the optimistic windows shown.

### Phase 8: First 2D World & Memory Upgrade
**Timeline:** May–August 2026 (3–4 months)  
**Compute:** Same TPU v5e-8 (still fits 1024 envs).  
**Goal:** Introduce spatial complexity while enhancing memory and navigation with AIF elements.  
**Key upgrades**  
- World: 32×32 → 64×64 grid using **CAX** library (ICLR 2025 oral — 10–100× faster 2D CA on TPU). All CA remains fully differentiable with `jax.remat` for memory safety.  
- Physics: Wireworld-style cellular automata (conductors, sources, transistors/gates, structural blocks).  
- Agents: 8–16 per universe, 9×9 local vision, 5-tile vocal radius.  
- Actions: Mine + Place (limited to conductors and basic gates). Core vaults still provide reliable reward.  
- Brain: **Small Mamba** (State Space Model) memory (linear scaling, fixed hidden state — mandatory for TPU compilation).  
- AIF Integration: Add AIF navigation system (adapted from de Tinguy et al., IWAI 2025) for real-time pathfinding; agents predict 3–5 steps ahead via forward generative models.  
- Fitness: Total energy harvested by the tribe + bonus for machine complexity (gate count).  
- Cultural transmission: Top 25 % agents’ memory embeddings seeded into newborns.  
**Success criteria**  
- Agents build simple circuits to reach cores.  
- Clear division of labor (scouts vs builders vs signal-relay specialists).  
- Reusable “blueprints” communicated via glyphs across the map.  
- Success rate >80 % on 500-step tasks.  
- AIF navigation produces emergent signal towers and better handling of occluded vaults.  
**Dependencies:** `pip install cax`; adapt de Tinguy et al. navigation module to JAX.

### Phase 9: Turing Sandbox – Open-Ended 2D
**Timeline:** September 2026–March 2027 (6–7 months)  
**Compute:** TPU v5e-8 or small pod slice.  
**Goal:** Achieve open-ended evolution with hierarchical multi-agent AIF and explicit curriculum.  
**Key upgrades**  
- World: Full **256×256** CAX grid with procedural biomes and scattered resource vaults (no pre-built solutions). Use `jax.remat` on rollouts.  
- Agents: 16–32 per universe.  
- Physics: Fully Turing-complete blocks (transistors, logic gates, etc.) using differentiable Neural CA.  
- LLM-in-the-loop: Your “God” translator runs on top 10 % every 250 generations → extracts dictionary → injects as auxiliary loss or embedding prior.  
- AIF Integration: Full hierarchical multi-agent AIF (VERSES-style high-level discrete skills for roles + low-level continuous control). Factorized generative models (Jaime et al., AAMAS 2025) for private beliefs about other agents.  
- Explicit Curriculum: Central “Adversarial Director” network (or CURATE-style module) whose loss is the agents’ learning progress. It generates puzzles exactly at the edge of current capability (Zone of Proximal Development) instead of pure random UED.  
- Fitness: Purely lifetime tribe energy harvested + complexity of persistent machines left on the map.  
- Open-endedness: Episodes run until starvation or 2,000 steps; new vaults spawn dynamically when mastery is detected.  
**Success criteria**  
- Emergent hierarchy, named agents, physical signal towers, and cultural traditions that survive generational turnover.  
- Agents invent and reuse complex machines across generations.  
- Vocabulary evolves into proto-grammar (conditional signals, planning statements).  
- Hierarchical AIF + curriculum achieves >60% success on multi-vault coordination tasks, outperforming pure RL baselines without pre-training.  
**Dependencies:** Fork VERSES “Mobile Manipulation” repo (arXiv:2507.17338) and Jaime et al. factorized models.

### Phase 10: Embodiment in High-Fidelity Simulation
**Timeline:** April–December 2027 (8–9 months)  
**Compute:** TPU pod + GPU cluster for sim rendering.  
**Goal:** Bridge to embodied robotics with AIF controllers for whole-body coordination and zero-shot adaptation.  
**Key upgrades**  
- Interface: Byte-grid vision (9×9 or 11×11) to JAX-native robotics simulators (MuJoCo, Brax, or Isaac Gym via JAX).  
- New modality: Audio (spoken commands become byte streams on the grid).  
- Tool-use: Special glyphs let agents call external JAX functions (math, search, code execution).  
- AIF Integration: VERSES/Fujii hierarchical controllers (IEEE 2025) with temporally hierarchical world models for stable whole-body control in sparse-reward sims.  
- Training stays fully in simulation (no physical hardware yet).  
**Success criteria**  
- Zero-shot sim-to-sim transfer across different robot morphologies.  
- Agents use tools and language to solve embodied tasks (navigation, manipulation, collaborative construction).  
- AIF controllers outperform RL in adaptation to noise/dynamics mismatch.  
**Dependencies:** `pip install mujoco brax isaacgym` (JAX wrappers); fork Fujii et al. repo.

### Phase 11–12: Recursive Self-Improvement & Multi-Agent Societies
**Timeline:** 2028–2029 (12–18 months total — frontier research horizon)  
**Compute:** Full TPU pod / Pathways cluster.  
**Goal:** Enable self-improvement in large-scale societies with AIF for decentralized coordination.  
**Key upgrades**  
- Meta-evolution: Agents propose and test small architecture mutations (NAS on Mamba/Transformer core).  
- Scale: 1,000–10,000+ agents across multiple universes with specialization and “trade”.  
- Environments: Indefinite runtime; new puzzles and biomes spawn automatically on mastery.  
- AIF Integration: Orchestrator-style monitoring (Beckenbauer et al., 2025) + agentic rulebooks (Constant et al., Frontiers 2025) for sustainable, decentralized coordination.  
- Full cultural evolution metrics (dictionary stability, tradition inheritance).  
**Success criteria**  
- Agents improve their own training loop or memory architecture.  
- Stable multi-agent societies with division of labor, markets, and long-term planning.  
- AIF enables efficient decentralized optimization without central control.  
**Dependencies:** Fork Beckenbauer Orchestrator and Constant rulebook repos.

### Phase 13+: True AGI Deployment
**Timeline:** 2029 onward (frontier research horizon)  
**Goal:** Deploy persistent, safe AGI societies with AIF priors for global stability and interpretability.  
**Key features**  
- Persistent multi-agent societies running 24/7 in simulation and real Spatial Web environments.  
- Grounded safety: Every action must reduce global free energy (active-inference prior).  
- LLM oracle access: Frontier models queried only through byte-level grounded channels.  
- AIF Integration: AIF safety priors (minimize global surprise) + Beckenbauer’s multi-agent reflection for ongoing optimization and self-revision in decentralized systems.  
- Global audit log: Your LLM translator becomes the permanent interpretability layer.  
**Success criteria**  
- Societies maintain stability in open-world deployments, adapting to real-time changes while minimizing surprise.  
- Safe, interpretable behaviors emerge from AIF priors, with auditable generative models for all decisions.  
**Dependencies:** Integrate with Spatial Web standards (IEEE 2874); use VERSES AXIOM for unified perception/planning/control in production.
