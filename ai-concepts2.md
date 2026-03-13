Based on the two repositories, here is a breakdown of the most important concepts, methods, and algorithms. These projects highlight two massive frontiers in AGI progress: **Multi-Agent Swarm Intelligence for complex world simulation** and **Autonomous AI Research (Recursive Self-Improvement)**.

### 1. MiroFish: Swarm Intelligence & Parallel World Simulation

**Repository:** `666ghj/MiroFish`
**Core Concept:** A multi-agent AI engine that ingests real-world data (news, policies, texts) to build a parallel digital sandbox. Thousands of autonomous agents with independent personalities and long-term memory interact to simulate future social, financial, or political outcomes.

**Key Concepts & Methods for AI/AGI:**

* **Swarm Intelligence & Emergence:** Moving beyond single-agent prompting, this explores how thousands of LLM-driven agents interacting in a confined environment produce emergent, unpredictable, but realistic macroscopic behaviors (e.g., market crashes, public opinion shifts).
* **Graph Retrieval-Augmented Generation (GraphRAG):** Used to build the foundational knowledge graph of the simulation.
* *Method Breakdown:* 1. **Extraction:** An LLM processes raw unstructured text to extract entities (nodes) and relationships (edges).
2. **Community Detection:** The graph is partitioned into hierarchical communities using algorithms like the Leiden algorithm.
3. **Summarization:** The LLM generates summaries for each community.
4. **Retrieval:** When an agent or the system queries information, it retrieves these structured community summaries rather than raw semantic text chunks, providing a much stronger "global" understanding of the simulated world.


* **Agentic Cognitive Architecture (Observation-Memory-Action Loop):** To simulate realistic humans, agents must have persistent state.
* *Algorithm/Method:* Based on the Generative Agents framework, an agent's memory retrieval determines its behavior.
* *Formula for Memory Retrieval Score:* $\text{Score} = \alpha \cdot \text{Recency} + \beta \cdot \text{Importance} + \gamma \cdot \text{Relevance}$
* *Recency:* Exponential decay based on how long ago the memory was formed.
* *Importance:* An LLM-assigned weight (e.g., buying a coffee = 1; witnessing a crime = 10).
* *Relevance:* Cosine similarity between the current situation's embedding and the memory's embedding.




* **Time-Series Dynamic Memory:** Agents dynamically update their memory structures as the simulation runs, allowing for "ReportAgents" to use tools to extract insights and generate future predictive reports based on the timeline.

### 2. autoRL: Autonomous AI Research over RL Environments

**Repository:** `harshbhatt7585/autoRL`
**Core Concept:** A fully autonomous "AI Scientist" loop designed to run over Reinforcement Learning (RL) environments. A coding LLM agent (like Codex/Claude/GPT-4) reads instructions, writes RL environments, tweaks hyperparameters, trains the model, and evaluates the results continuously without human intervention.

**Key Concepts & Methods for AI/AGI:**

* **Autonomous Research Loop (LLM-as-a-Researcher):** This represents a massive step toward AGI recursive self-improvement.
* *Method Breakdown:*
1. **Hypothesis/Code Generation:** The LLM modifies `candidate/env.py` (the task) and `candidate/train.py` (the hyperparameters).
2. **Execution:** The system runs the fixed evaluator (using Simverse).
3. **Scoring:** The environment is scored strictly by `mean_eval_return` across multiple seeds and episodes.
4. **Feedback & Iteration:** The LLM reads the generated `results.tsv` and error logs. If the agent failed to learn, the LLM deduces why, writes new code, and kicks off the next experiment.




* **Fixed Evaluator Design Principle:** To allow an AI to do valid research, the outer boundaries must be rigid. autoRL deliberately freezes the scoring metric and runtime to ensure the LLM doesn't "cheat" the evaluation system (e.g., by making the metric artificially easy instead of solving the environment).
* **Proximal Policy Optimization (PPO):** The underlying reinforcement learning algorithm used as the fixed evaluator to train the environments built by the AI.
* *Algorithm/Formula:* PPO is an actor-critic method that limits how much the policy can change in a single update, ensuring stable learning. It maximizes a clipped surrogate objective:
$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right]$
* $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$ is the probability ratio of the new policy over the old policy.
* $\hat{A}_t$ is the estimated advantage (how much better the action was than expected).
* $\epsilon$ is a hyperparameter (usually 0.2) that clips the ratio, preventing destructively large updates.





### Why these matter for AGI Progress

1. **MiroFish** tackles the **world-modeling and simulation** problem. For an AGI to be safe and useful, it must be able to simulate the complex, chaotic second-order effects of its actions on human society before it takes them.
2. **autoRL** tackles **recursive capability scaling**. Human researchers are currently the bottleneck in AI progress. A framework that allows an LLM to automatically spin up, train, debug, and optimize Reinforcement Learning agents continuously is the exact mechanism required for an intelligence explosion.
