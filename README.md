# byte-agi

I had help from generative AI (Gemini and xAI) to come up with this text and the project concepts and code. Almost all text/code is AI generated. The hope for the project is to create all of the below phases and test them, and for the earlier phases to be runnable for free within a single Kaggle notebook with the GPU P100 selected on the right-side menu. Not there yet. 

byte-agi is a minimal, biologically-plausible, mathematically-grounded sandbox for emergent intelligence.  

Everything is a single byte (0–255). Vision is blind to self. Speech is visual (your body literally changes glyph). The only objective is Expected Free Energy (FEP/EFE). The code stays in pure JAX/Flax so every line compiles to GPU/TPU and scales with `vmap`/`pmap`. No Unity, no separate audio stream, no hand-crafted rewards or exploration bonuses—only physics, metabolism, curiosity, and death.  

This is a clean launchpad that lets language, cooperation, cultural transmission, and open-ended tool use emerge.

### Phase 1: Hello World – Berry Hunter (Single Agent, Trainable)

* 16-byte 1D tape, one `@` agent, one Berry.
* Blind-self vision: 9-tile window (4 left + 4 right), center masked to EMPTY.
* Proprioception input: concatenate current hunger + last action + last vocal (even if Phase 1 vocal=0).
* Brain: `nn.Embed(256, hidden_dim)` → LSTMCell → policy (3 actions) + predicted next 9-tile vision.
* Full episode rollouts via `jax.lax.scan` (50–100 steps) so gradients flow end-to-end.
* Action sampling: categorical + small entropy bonus.
* Real hunger decay + regeneration (agent must keep eating).
* Objective: simple extrinsic Free Energy (hunger error). Train with real gradients.
* Success: agent reliably finds and eats berry; Free Energy visibly → 0.
* *Debug output: Tick | Pos | Hunger | Total_FE + live tape render.*

### Phase 2: Stag Hunt – Visual Language Birth & Vectorization

* Expand to 32-byte tape. Add Button, Door, Core.
* Agents now output vocal (0=silent `@` or 1–26=A–Z). Speaking changes their body byte and applies 0.01 metabolic tax.
* Same blind 9-tile vision + proprioception (now includes own energy).
* Brain: policy_logits (3) + vocal_logits (27) + predicted_vision head.
* Add early `jax.vmap`: Vectorize over 1,024 parallel universes (8 agents each) before implementing Phase 3. Doing this forces us to use `jnp.where` instead of Python `if/else` statements, ensuring the framework scales cleanly without requiring a massive code rewrite when the complex epistemic math is added later.
* Scripted demo first, then train both agents jointly.
* Success metric: one agent learns to step on button and flash a consistent capital letter; the other sees it in its local view and walks through the opened door to the Core.
* *Debug output: Rendered tape shows exactly the “X opens door” moment.*

### Phase 3: Deep Active Inference – Curiosity & Imagination

* Split brain into Encoder → Transition Model (variational) → Policy.
* Full Expected Free Energy (EFE) loss:
* Extrinsic = hunger/energy error.
* Epistemic = KL(predicted next vision || actual next vision) after one-hot encoding the 9 tiles.
* Total_FE = extrinsic + β·epistemic (β anneals from high to low).


* Transition model trained with cross-entropy predictive coding (the agent literally learns to “imagine” the next tape state).
* No hand-coded exploration—the math forces pure epistemic wandering until the map is known.
* Success: beautiful loss curves (epistemic high at spawn → drops to zero; pragmatic takes over). Agents explore unknown tiles then exploit.

### Phase 4: Emergent Physics – Cellular Automata Replace All Hard-Coding

* Replace every button/door/core rule with local Rule-110-style cellular-automaton updates (only 3–5 bytes interact locally).
* Agents must discover crafting, doors, and resources through experiment.
* Keep metabolic tax and blind vision.
* Language analysis suite (logged automatically): vocal bigram/trigram frequencies, mutual information (vocal ↔ button state), convention success rate.
* Success: Pareto distribution in vocabulary (structured language emerges, not random flashing). Agents invent new physics uses.

### Phase 5: Evolution & Cultural Transmission – Generational Turnover

* Every 500 steps: kill agents below energy threshold.
* Birth new agents by copying survivor weights + small Gaussian mutation (or zero-init for pure cultural reset).
* Lamarckian Evolutionary Mechanics: Because the agents are learning via gradient descent during their lifetime, we are simulating the Baldwin Effect. Newborns inherit the visual 26-cap language from parents via observation only—no weight transfer required for culture to spread.
* Continue vmap scaling and EFE training across generations.
* Success: the “X = open door” convention (or better ones) survives and refines across generations. You literally watch cultural evolution in the vocab logs.

### Phase 6: The AGI Runway – Hyperscale & External Translation

* Distributed training: `jax.pmap` across multi-GPU/TPU cluster (same code, instant scaling).
* Integrate L-Mul (Linear-complexity Multiplication from BitEnergy AI): replace FP multiplies in the LSTM core with integer adds → billions of parameters at extreme efficiency.
* The 1D Constraint: Keep the environment strictly 1D until everything above works perfectly. 2D pathfinding consumes 90% of a neural network's capacity in standard RL. Restricting it to 1D ensures 100% of the compute is spent on language, logic, and theory of mind. Optional 2D upgrade (16×16 grid with 7×7 local view) serves as the ultimate final test of generalization.
* “God” LLM translator: every 10 steps feed the last 32 bytes + vocal history to an external LLM and ask it to translate the emerging 26-cap language into English. Purely for human insight.
* Open-source license: MIT + "research use encouraged" so labs / academia can freely play.
