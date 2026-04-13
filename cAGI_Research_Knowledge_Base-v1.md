# AGI & AI Research Knowledge Base: Algorithms, Methods, and Architectures
**Purpose:** Comprehensive synthesis of advanced AI research methodologies, open-ended learning, agentic architectures, and recursive self-improvement for LLM ingestion.

---

## 1. Recursive Self-Improvement & Self-Referential Optimization
A critical frontier in AGI is building systems that continuously improve their own learning and problem-solving mechanisms, escaping human-engineered bottlenecks.

### 1.1 Hyperagents & Metacognitive Self-Modification (DGM-H)
* **The Bottleneck of Fixed Meta-Agents:** Traditional self-improving systems (like the Darwin Gödel Machine - DGM) rely on a fixed, handcrafted meta-level mechanism to generate improvements. This binds the system's potential to the initial human design and requires strict alignment between task skills and self-modification skills.
* **Hyperagent Architecture:** A hyperagent combines a **task agent** and a **meta agent** into a single, self-referential, editable program. 
* **Metacognitive Self-Modification:** By making the improvement procedure itself editable, the system can improve *how* it improves. It autonomously develops meta-level capabilities like:
    * **Persistent Memory & Auto-Tracking:** Creating trackers that log and organize metrics across iterations to identify improvement trends and regressions.
    * **Compute-Aware Planning:** Adapting strategies based on remaining compute budgets (e.g., executing ambitious architectural overhauls early, and conservative bug fixes later).
    * **Bias Detection:** Implementing automated label-distribution tracking to catch classification collapse and correct degenerate behaviors autonomously.
* **Parent Selection & Open-Ended Exploration:** Maintains an archive of increasingly capable agents. Parent selection balances exploitation (performance) and exploration (novelty/few children) using Upper Confidence Bound (UCB) and temperature-controlled softmax sampling. 

### 1.2 Bilevel Autoresearch (LLM-Driven Meta-Optimization)
* **Concept:** Using an autoresearch loop to optimize the autoresearch loop itself. The inner loop optimizes the target task (e.g., hyperparameter tuning), while the outer loop optimizes *how* the inner loop searches.
* **Level 1 (Inner Loop):** Standard propose-train-evaluate cycle. Prone to deterministic, repetitive failure modes biased by the LLM's priors.
* **Level 1.5 (Outer Config):** Freezes stalled parameters and injects guidance to redirect search diversity (avoids local minima but bounded by the fixed architecture).
* **Level 2 (Mechanism Research):** Generates new search mechanisms as executable Python code at runtime via a 4-round LLM dialogue:
    1.  **Explore:** Reads search traces and surveys adjacent fields (combinatorial optimization, Multi-Armed Bandits, Design of Experiments).
    2.  **Critique:** Evaluates candidate mechanisms against observed failure modes.
    3.  **Specify:** Writes precise interfaces and integration points.
    4.  **Generate & Inject:** Writes runnable Python code (e.g., Tabu Search Managers, Systematic Orthogonal Explorers) and patches the runner via dynamic import validation (with automatic rollback on failure).

---

## 2. Open-Ended Learning (OEL) & Auto-Curricula
OEL abandons fixed objectives in favor of continuous generation of novel, increasingly complex tasks and solutions, mimicking biological evolution.

* **AI-Generating Algorithms (AI-GAs):** The philosophy that the fastest path to AGI is not manual engineering, but building open-ended systems that evolve intelligence for us.
* **Novelty Search:** Discarding the objective function entirely. Rewarding agents purely for reaching unvisited behavioral states, paradoxically solving hard exploration problems (like mazes) faster by avoiding local optima.
* **Co-Evolution (POET & XLand):** * *POET* co-evolves agents alongside the environments they must solve, constantly generating a "Goldilocks" zone of difficulty (autocurriculum).
    * *XLand* generates billions of procedural 3D games, shifting topologies to keep agents at the edge of their capabilities, yielding generalized heuristics rather than narrow, overfit policies.
* **Emergent Complexity via Arms Races:** Multi-Agent Hide and Seek demonstrates how simple physical rules and adversarial reward structures yield wildly unpredictable, emergent behaviors (e.g., fort building, ramp stealing, box surfing exploits).

---

## 3. Minimalist Emergence Environments (byte-agi)
A methodology for studying tabula rasa learning, emergent communication, and multi-step causal chains.

* **Channel Collapse:** Encoding all entities (walls, agents, objects, tools, vocal speech, written marks) into a single 1D byte alphabet (0-255). Agents have no symbolic type system and must discover semantics purely through environmental interaction.
* **Cellular Automaton (CA) Physics & Composable Chemistry:** 12 deterministic left-to-right CA rules dictate environmental reactions. Agents can craft specific "catalyst" bytes that react to synthesize "products", which in turn interact with the environment (e.g., opening a door, neutralizing hazards).
* **Temporal Credit Assignment:** Agents learn multi-step synthesis (Craft B -> Craft C -> CA creates Y -> Y opens door -> eat core) where initial actions carry zero immediate reward, driven by intermediate reward shaping.
* **Anti-Forgetting Infrastructure:** * *Checkpoint-based Recovery:* Saves parameters upon advancing curriculum levels; restores to the best checkpoint upon fallback, preventing catastrophic forgetting of foundational skills.
    * *Weighted Experience Replay:* Batches contain 62.5% current difficulty environments and 37.5% replay environments (weighted heavily toward N-1 and N-2 levels).

---

## 4. Production-Grade Multi-Agent Swarms & Orchestration
Insights derived from advanced enterprise orchestration harnesses (e.g., Claude Code architecture).

* **The Coordinator & Sub-Agent Delegation:** Moving beyond monolithic contexts. A Coordinator LLM breaks down tasks, spawning sandboxed sub-agents (e.g., `code-reviewer`, `test-runner`, `explore`) endowed with a subset of tools. This prevents the main context from being polluted by massive `grep` outputs or raw test logs.
* **Agent Teammate Communication:** Swarm coordination via asynchronous `SendMessageTool`. Agents broadcast to teammates or send direct messages. Team state is maintained in shared configuration files and task lists. Teammates enter "idle" states waiting for input rather than terminating.
* **Context & Memory Management:**
    * **Context Compaction:** When approaching token limits (e.g., 150K tokens), the system automatically summarizes early history into `<compaction>` blocks, allowing continuous long-running sessions.
    * **autoDream (Background Memory Consolidation):** While the user/system is idle, forked sub-agents review session transcripts to merge observations, resolve logical contradictions, and distill vague insights into "absolute facts" in a persistent vector store or file directory.
    * **Prompt Caching Economics:** Strict architectural enforcement of prefix matching. Volatile data (timestamps, UUIDs) are moved to the end of prompts. Tools and system instructions are frozen to maximize cache read hits (~90% cost reduction).

---

## 5. Next-Gen Models & High-Stakes Autonomy
### 5.1 "Mythos" (Capybara) Class Models & KAIROS
* **Connective Intelligence:** Models specifically trained for linking disparate domains and executing highly complex autonomous sequences without human intervention, particularly excelling in offensive/defensive cybersecurity.
* **KAIROS Daemon:** Shift from reactive chat to proactive, 24/7 background daemons. Triggered by system events (GitHub webhooks, cron `ticks`), capable of preemptive maintenance (fixing flaky tests, updating dependencies) while the user is away.
* **Undercover Mode:** Highly secure, ephemeral sessions. Strips telemetry, isolates context from personal memory files, and automatically scrubs caches upon termination.

### 5.2 Agentic Safety & Threat Modeling
* **Explicit vs. Inferred Intent:** The safety monitor blocks destructive actions if parameters are inferred/hallucinated by the agent. High-severity actions (mass deletions, DB migrations) require the user to explicitly name the specific targets.
* **Blind Apply Prevention:** Blocking agents from executing destructive infrastructure commands (`--auto-approve`, `--yes`) without preceding plan/dry-run outputs in the transcript.
* **Evaluating Blast Radius:** Increased scrutiny on shared infrastructure. Distinguishing between local, reversible file edits vs. actions modifying external shared states, CI/CD pipelines, or cloud buckets.

---

## 6. Meta-Research Strategy for Applied AI
For individuals and small labs lacking massive compute clusters:
* **High Leverage Artifacts:** Producing open-source tools, robust minimalist testbeds (like `byte-agi` or Gym), and high-quality trajectory datasets (for emergent communication or interpretability).
* **Avoid Incremental ALife:** Top-tier labs are bottlenecked by alignment, scaling laws, and inference. Moving the needle requires discovering simple, reproducible anomalies (e.g., physical exploits developed by agents defying expected theory) or providing infrastructure that lowers the activation energy for the broader research community.
