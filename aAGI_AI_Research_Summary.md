# Comprehensive AGI & AI Research Knowledge Base (2026 Edition)

**Purpose:** A detailed, LLM-optimized compilation of the most critical concepts, algorithms, frameworks, and methodologies driving artificial general intelligence (AGI) and AI research, spanning multi-agent evolution, code superoptimization, and physical intuition models.

---

## 1. The Unified AGI Flywheel
The "Unified AGI Flywheel" represents a closed-loop, compounding system architecture consisting of four continuously running layers [cite: 284]:
* **Emergence Layer:** Involves agents evolving language and skills from a blank slate (tabula rasa) within simulated environments running at millions of steps per second [cite: 286, 287]. Methods like Multi-Agent PPO (MAPPO) and Gumbel-Softmax are utilized here [cite: 287].
* **Grounding & World-Model Layer:** Focuses on training latent world models on the trajectories generated in the emergence layer [cite: 289]. It leverages the Expected Free Energy objective for planning and creates persistent interactive simulators [cite: 290, 291].
* **Self-Evolution Layer:** Utilizes Group-Evolving Agents (GEA) that can autonomously rewrite their own memory, tools, prompts, and code from episode transcripts [cite: 293, 294]. Safety is maintained via verifiers and canary sandboxes [cite: 294].
* **Test-Time Scaling & Refinement Layer:** Allocates variable compute at inference time [cite: 297]. It employs guided search methods, potential landscapes (SLOPE), and program synthesis retries to achieve superhuman performance [cite: 297, 298].

**The Flywheel Equation:** Progress is defined as the product of the four stages: `Progress(t+1) = f(Emergence(t) * WorldModel(t) * SelfEvolution(t) * TestTimeCompute(t))` [cite: 300, 301]. Because each iteration re-uses discoveries from previous agents, the progress compounds exponentially [cite: 302].

## 2. AlphaEvolve: Code Superoptimization & Algorithm Discovery
AlphaEvolve is a coding agent developed by Google DeepMind that orchestrates an autonomous pipeline of LLMs to improve algorithms by making direct changes to the code [cite: 634, 635].
* **Scale and Flexibility:** Unlike earlier models (like FunSearch) that evolved single functions, AlphaEvolve can evolve entire code files spanning hundreds of lines [cite: 664, 666]. It utilizes automated evaluation metrics to assess and score generated solutions [cite: 656].
* **LLM Ensemble:** The system leverages an ensemble of Gemini 2.0 Flash (for low-latency, high-throughput candidate generation) and Gemini 2.0 Pro (for highly capable, breakthrough suggestions) [cite: 873, 874, 875, 876, 877].
* **Distributed Evolutionary Database:** It balances exploration and exploitation using an evolutionary database inspired by MAP elites and island-based population models [cite: 899, 900].
* **Diff-Based Mutations:** When modifying large codebases, AlphaEvolve generates changes via targeted `<<<<<<< SEARCH` and `>>>>>>> REPLACE` diff blocks to ensure targeted updates to specific codebase segments [cite: 865, 866, 868, 871].
* **Breakthrough Achievements:**
    * **Matrix Multiplication:** AlphaEvolve discovered a novel algorithm that multiplies two 4x4 complex-valued matrices using only 48 scalar multiplications, breaking the 56-year-old record set by Strassen's algorithm (which required 49) [cite: 640, 641, 675].
    * **Data Center Scheduling:** Discovered a simplified, effective heuristic function for Google's Borg scheduler, successfully recovering an average of 0.7% of Google's fleet-wide compute resources [cite: 1088, 1093, 1099].
    * **Compiler and Hardware Optimization:** Optimized XLA-generated intermediate representations for FlashAttention kernels (yielding a 32% speedup) and refined Verilog circuit designs for TPU arithmetic units [cite: 1141, 1147, 1157, 1163].
    * **Mathematics:** Found novel, provably correct constructions that surpassed the State-Of-The-Art (SOTA) in about 20% of evaluated mathematical problems, including kissing numbers in 11 dimensions and the minimum overlap problem [cite: 678, 679].

## 3. Physical Intuition & World Modeling (V-JEPA)
Models must learn the physical dynamics of the real world. Meta's V-JEPA (Video Joint Embedding Predictive Architecture) learns intuitively from video without explicit physical assumptions [cite: 2138, 2139].
* **Latent Space Prediction:** Traditional models operate in "pixel space," predicting exact masked pixels, which forces them to waste capacity on irrelevant details (like the motion of leaves) [cite: 2147, 2150]. V-JEPA masks portions of video frames but predicts missing information in a highly abstracted "latent" representation space instead [cite: 2158, 2159].
* **Architecture:** It utilizes an Encoder 1 (processes masked frames), an Encoder 2 (processes unmasked frames), and a Predictor (takes the latent output of Encoder 1 and predicts the latent output of Encoder 2) [cite: 2164, 2166, 2167].
* **Quantifiable "Surprise":** When tested on physically impossible events (such as an object failing to reappear after passing behind a barrier), the model's prediction error sharply spikes [cite: 2182, 2183, 2184, 2185]. This demonstrates an emergent understanding of object permanence [cite: 2176].
* **V-JEPA 2:** A 1.2-billion-parameter successor pretrained on 22 million videos [cite: 2192]. It has been fine-tuned using just ~60 hours of robot data to successfully plan robotic actions [cite: 2193]. However, its memory horizon is limited to a few seconds, requiring future architectures to address "goldfish" memory limitations [cite: 2196, 2198].

## 4. The KISS AGI Framework (byte-agi)
A mathematically-grounded, biologically-plausible sandbox for emergent intelligence that compiles entirely to GPUs/TPUs via JAX/Flax [cite: 480, 482].
* **Fundamental Constraints:** Everything in the environment is represented by a single byte (0-255) [cite: 481, 539]. The vision system is blind to the agent's own self, and speech acts as a visual change to the agent's body byte [cite: 481, 540].
* **Objective Function (Expected Free Energy):** The only objective is Expected Free Energy (EFE) [cite: 482, 541].
    * `Total_FE = extrinsic + β * epistemic` [cite: 512, 564].
    * *Extrinsic* targets hunger and metabolic energy error [cite: 510, 562].
    * *Epistemic* targets the KL divergence between the predicted next visual state and the actual next visual state, mathematically forcing pure exploration/curiosity without hand-coded rewards [cite: 511, 514, 563, 566].
* **Brain Architecture:** Embeddings route into an LSTMCell (or Mamba) policy, predicting actions and the next 9-tile vision state via a variational Transition Model [cite: 489, 508, 547]. 
* **Vectorization & Scaling:** Utilizing `jax.vmap` allows the framework to scale to 1,024 parallel universes instantly, maintaining high throughput (>50k steps/sec on a single GPU) [cite: 520, 557].
* **Cellular Automata Physics:** Hardcoded environment rules (doors, buttons) are eventually replaced by local Rule-110-style cellular-automaton updates, requiring agents to discover physics and crafting emergently [cite: 518, 519, 568].
* **Evolution & Lamarckian Mechanics:** Agents learn via gradient descent during their lifetime (the Baldwin Effect) [cite: 574]. Newborn agents copy survivor weights with Gaussian mutations and inherit visual languages via observation (cultural transmission) [cite: 525, 526, 575].

## 5. Critical Algorithms & Mathematical Formulations
* **Language Model Zero (LM Zero):** Shifts from predicting the next token (imitating human data) to utilizing multi-agent MARL where language evolves purely as an optimal control policy [cite: 115].
    * `Reward (r_t) = I(achievement flag) - λ * (talk cost)` (An energy penalty enforces an information bottleneck) [cite: 118].
* **Gumbel-Softmax:** Used to evolve discrete communication tokens via backpropagation [cite: 140]. 
    * `y_i = exp((g_i + log(pi_i))/tau) / sum(exp((g_j + log(pi_j))/tau))` [cite: 141].
* **Multi-Agent PPO (MAPPO):** Uses Centralized Training with Decentralized Execution (CTDE) where a centralized critic sees a global view while actors only see local observations and messages [cite: 148].
* **PBRS & SLOPE:** Shifts scalar regression to potential landscapes with optimistic upper bounds [cite: 152].
    * `r~(s,a,s') = r(s,a,s') + γ * Φ(s') - Φ(s)` [cite: 154]. SLOPE applies asymmetric weighting on upper-quantile returns to amplify rare successes, providing dense gradients for planning [cite: 153, 155].
* **Test-Time Scaling (o1 Paradigm / MPPI):** Scales test-time inference compute to deliberately search for answers using Chain-of-Thought (CoT) and Process Reward Models (PRMs) [cite: 195, 201]. Expected return is modeled as `J = max(E[majority vote or search(k samples)])` proportional to FLOPs [cite: 199].

## 6. Emerging AGI Paradigms (2024-2026)
* **Communicating Plans, Not Percepts:** Agents utilize an Intention Trajectory Grounding Module (ITGM) to share imagined policy trajectories rather than raw observations, resulting in highly compact intention-level protocols [cite: 210, 211].
* **Superhuman Adaptable Intelligence (SAI):** The true target for AGI is framed as fast learning and zero-shot transfer across novel tasks, heavily relying on JEPA-style world models over purely human-like general emulation [cite: 326, 328].
* **Cost to Verify vs. Cost to Automate:** As the cost to automate falls exponentially, human interaction transitions strictly into high-value verification, making immutable kernels and Proposer-Verifier loops non-negotiable for AI safety [cite: 332, 333, 335].
