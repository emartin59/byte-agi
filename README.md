# byte-agi

I had help from generative AI (Gemini and xAI) to come up with this text and the project concepts and code. Almost all text/code is AI generated. The hope for the project is to create all of the below phases and test them, and for the earlier phases to be runnable for free within a single Kaggle notebook with the GPU P100 selected on the right-side menu. Not there yet.

byte-agi is a minimal, biologically-plausible, mathematically-grounded sandbox for emergent intelligence.

Everything is a single byte (0–255). Vision is blind to self. Speech is visual (your body literally changes glyph). The only objective is Expected Free Energy (FEP/EFE). The code stays in pure JAX/Flax so every line compiles to GPU/TPU and scales with `vmap`/`pmap`. No Unity, no separate audio stream, no hand-crafted rewards or exploration bonuses—only physics, metabolism, curiosity, and death.

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
* **Open-Source License:** MIT + "research use encouraged" so labs / academia can freely play.
