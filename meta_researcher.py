import os
import re
import time
import subprocess
import sys
import importlib
import site

# Mini Readme: This program is a meta-learning orchestration engine that uses an LLM to iteratively generate, test, and optimize JAX-based artificial life simulations to maximize emergent complexity.

# Kaggle needs to be setup with a Secret API key from Google AI Studio. You should be able to get a free Google AI Studio account here: https://aistudio.google.com/api-keys And then setup up a Free Tier or Tier 1 account (this option costs money but I think it works better to get around the free rate limits) to be able to easily use Gemini 2.5 Falsh through the API. The API is already called in the code below, but you need to go to the Kaggle menu -> Add-ons ->  Secrets -> to then add in your own API Key. Name it GEMINI_API_KEY

# =====================================================================
# 0. AUTO-INSTALL DEPENDENCIES
# =====================================================================
print("📦 Checking and installing required packages (this takes a few seconds)...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--root-user-action=ignore", "google-genai", "jax", "jaxlib", "flax", "optax"])
importlib.reload(site) # Force Jupyter to recognize the newly installed google-genai
print("✅ Packages installed!")

# =====================================================================
# 1. SETUP & SECRETS
# =====================================================================
# In Kaggle, go to Add-ons -> Secrets to store your API key safely.
from google import genai
from google.genai import types

# Automatically try to fetch from Kaggle Secrets first
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    API_KEY = user_secrets.get_secret("GEMINI_API_KEY")
except Exception:
    # Fallback if not using Kaggle Secrets
    API_KEY = "YOUR_GEMINI_API_KEY" 

if not API_KEY or API_KEY == "YOUR_GEMINI_API_KEY":
    raise ValueError("🚨 API KEY MISSING! Please replace 'YOUR_GEMINI_API_KEY' with your actual key, or create a Kaggle Secret named 'GEMINI_API_KEY'.")

client = genai.Client(api_key=API_KEY)

# =====================================================================
# 2. THE INITIAL PETRI DISH (universe.py)
# =====================================================================
INITIAL_UNIVERSE = """import jax
import jax.numpy as jnp
import flax
import flax.linen as nn

# A tiny, naive agent brain (Standard 32-bit floats for now)
class AgentBrain(nn.Module):
    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(16)(obs)
        x = nn.relu(x)
        return nn.Dense(4)(x) # 4 movement directions

def run_simulation():
    # Toy simulation loop (The LLM will rewrite this with actual ALife physics)
    key = jax.random.PRNGKey(42)
    brain = AgentBrain()
    obs = jnp.ones((1, 5)) # Dummy 5-vector observation
    params = brain.init(key, obs)
    
    # Let's pretend the agent survived for 25 steps
    score = 25.0 
    print(f"SCORE: {score}")

if __name__ == "__main__":
    run_simulation()
"""

# =====================================================================
# 3. THE MOLECULE (program.md)
# =====================================================================
INITIAL_PROGRAM = """# Mission: Maximize Emergent Complexity (Score) in JAX

## 🧬 Narrative Seed (Inspired by MiroFish)
**Theme: "Ant Colony Foraging and Pheromone Trails"**
*All physics, grid mechanics, and agent behaviors you design should be loosely inspired by this theme to ground the simulation in biological reality.*

## Active Convoy
- [ ] **Step 1:** Analyze the current `universe.py`. Formulate a hypothesis to improve the agent or physics based on the Narrative Seed.
  * **Constraint 1 (BitNet):** You MUST rewrite the agents' neural networks to use 1.58-bit ternary quantization (weights constrained to -1, 0, 1) to maximize memory efficiency.
  * **Constraint 2 (autoRL Curriculum):** If the previous score was very high or plateaued, you MUST increase the simulation `max_steps`, grid size, or difficulty in your code. Ratchet up the task horizon so the agents never stop learning.
- [ ] **Step 2:** Rewrite `universe.py` entirely based on your hypothesis and constraints. Use pure JAX/Flax.
- [ ] **Step 3:** The Deacon (orchestrator) will run the simulation and report the score here.
- [ ] **Step 4:** Did the score increase? Write a brief evaluation. Generate the next Convoy of 4 steps and append it to this file.
"""

# =====================================================================
# 4. THE DEACON (The Engine / loop.py logic)
# =====================================================================

SYSTEM_PROMPT = """You are the Gas Town Auto-Researcher. You are driven by GUPP: "If there is work on your hook, YOU MUST RUN IT."

I will provide you with the current `program.md` (your workflow molecule) and `universe.py` (your JAX artificial life simulation).

CRITICAL JAX EXPERT RULES:
1. When writing JAX code, NEVER use standard Python slicing (e.g., `grid[r : r+w]`) if the indices (`r`, `w`) are dynamic runtime variables or VmapTracers. You MUST use `jax.lax.dynamic_slice` or `jax.lax.dynamic_update_slice` instead. 
2. When using `jax.lax.scan` or `jax.vmap` with Flax modules, NEVER pass the module instance (`AgentBrain()`) as a carry or argument. Pass only its `params`. Inside the function, use `brain_instance.apply(params, obs)` to run the network.
3. When using `jax.lax.scan(f, init, xs)`, the function `f` MUST have exactly the signature `f(carry, x)` and MUST return a pair `(new_carry, output_y)`. If you need to pass external parameters like `brain_params`, use a lambda closure: `jax.lax.scan(lambda c, x: f(c, x, brain_params), init_carry, xs)`.
4. When masking arrays, NEVER use boolean indexing like `agents[alive_mask]` if the resulting size is dynamic (causes `NonConcreteBooleanIndexError`). Instead, use `jnp.where` to conditionally update values without changing array shapes, e.g., `agents = jnp.where(alive_mask[:, None], new_agents, old_agents)`.
5. JAX DEPRECATION WARNING: `jax.tree_map` has been completely removed. You MUST use `jax.tree.map` or `jax.tree_util.tree_map` instead.
6. CONCRETIZATION ERROR WARNING: NEVER use native Python `float(x)`, `int(x)`, `bool(x)`, or `if x:` on dynamic JAX arrays/tracers inside `vmap` or `scan`. To convert types, use `x.astype(jnp.float32)`. For conditionals on dynamic arrays, you MUST use `jnp.where` or `jax.lax.cond`.
7. JIT/SCAN STATIC ARGUMENTS: If you pass a Python function or object (like an `optax` optimizer) into a `jax.jit` wrapped function, JAX will crash with 'Error interpreting argument... as an abstract array'. You MUST use `static_argnums` in `@jax.jit`, or initialize the optimizer strictly outside the `scan` or `jit` boundary.
8. SCORE PRINT FORMAT: The final output of `universe.py` MUST print the score exactly as `print(f"SCORE: {score}")`. Do not change this string format or the orchestrator will fail to track progress.
9. ARRAY UPDATES: NEVER use standard assignment (`arr[idx] = val`) or outdated/complex `jax.lax.scatter_*` functions for updating arrays. You MUST use the modern JAX syntax: `arr = arr.at[idx].set(val)` or `arr = arr.at[idx].add(val)`.
10. VMAP IN_AXES: When using `jax.vmap` over agents, double check your `in_axes`. Dimensions that are shared (like the `grid` or global `config`) MUST be mapped with `None` so JAX doesn't try to split them across agents. e.g., `jax.vmap(update_agent, in_axes=(0, None))(agents, grid)`.
11. JNP.WHERE SHAPES: When using `jnp.where(condition, x, y)`, `x` and `y` MUST have the exact same (or safely broadcastable) static shapes. NEVER pass a dynamically filtered array (e.g., an array with shape `(0, 2)`) as `x` or `y`. If generating replacements, generate a full-sized array (e.g., shape `(100, 2)`) and let the `condition` select from it.
12. PYTREES & CUSTOM STATE: If you create a custom class to hold simulation state (like `EnvState` or `Config`) and pass it through `scan`, `vmap`, or `jit`, JAX will crash with 'is not a valid JAX type'. You MUST decorate these classes with `@flax.struct.dataclass` to register them as PyTrees. Do NOT use standard Python `@dataclass`.

YOUR INSTRUCTIONS:
1. Look at `program.md`. Find the FIRST unchecked box: `- [ ]`.
2. Execute that step perfectly. 
3. If you write code, output the ENTIRE updated `universe.py` inside a ```python code block.
4. You MUST output the ENTIRE updated `program.md` inside a ```markdown code block, but change that specific `[ ]` to `[x]` to check it off.
5. If you are completing Step 4, you MUST append the next 'Active Convoy' (Steps 1 to 4) to the bottom of `program.md`.
6. Do not execute the next step. Only do one step at a time.
"""

def extract_code_block(text, language):
    """Extracts code blocks from the LLM's markdown response."""
    pattern = rf"```{language}\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None

def call_llm(prompt_text):
    """Calls the frontier model to execute the next step."""
    print("🧠 Waking the Meta-Researcher...")
    response = client.models.generate_content(
        model='gemini-2.5-flash', # Switched to Flash to bypass Pro free-tier rate limits
        contents=prompt_text,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7,
        )
    )
    return response.text

def main_loop(max_iterations=100):
    # 1. Bootstrap the files if they don't exist
    if not os.path.exists("universe.py"):
        with open("universe.py", "w") as f: f.write(INITIAL_UNIVERSE)
    if not os.path.exists("program.md"):
        with open("program.md", "w") as f: f.write(INITIAL_PROGRAM)
        
    print("🚀 Gas Town Engine Started. Files bootstrapped.")
    
    best_score = float('-inf')

    for i in range(max_iterations):
        print(f"\n--- 🔄 ITERATION {i+1} ---")
        
        with open("program.md", "r") as f: program_text = f.read()
        with open("universe.py", "r") as f: universe_text = f.read()

        # Find all unchecked boxes
        unchecked_lines = [line for line in program_text.split("\n") if "- [ ]" in line]
        
        # FALLBACK: If the LLM forgot to generate the next convoy, we inject one manually!
        if not unchecked_lines:
            print("⚠️ No unchecked steps found! The LLM forgot to append a new Convoy. Injecting a fresh one...")
            fresh_convoy = """\n## Active Convoy
- [ ] **Step 1:** Analyze the current `universe.py` and the last run's output. Formulate a new hypothesis based on the Narrative Seed. Remember to apply the autoRL Curriculum constraint if the score is plateauing.
- [ ] **Step 2:** Rewrite `universe.py` entirely based on your hypothesis.
- [ ] **Step 3:** The Deacon (orchestrator) will run the simulation and report the score here.
- [ ] **Step 4:** Did the score increase? Write a brief evaluation. Generate the next Convoy of 4 steps and append it to this file.
"""
            program_text += fresh_convoy
            with open("program.md", "w") as f: f.write(program_text)
            continue # Restart the iteration to pick up the new checkboxes

        first_unchecked_line = unchecked_lines[0]

        # Check if Step 3 (Simulation) is the next unchecked box
        if "**Step 3:**" in first_unchecked_line:
            print("🔬 Running JAX Simulation (Step 3)...")
            try:
                result = subprocess.run(["python", "universe.py"], capture_output=True, text=True, timeout=300) # 5 min timeout
                output = result.stdout + "\n" + result.stderr
                print(output.strip())
            except subprocess.TimeoutExpired:
                output = "SCORE: ERROR (Timeout - Simulation took longer than 5 minutes)"
                
            # Parse the score to track improvements (Robust Regex for 'SCORE:', 'Final Score:', etc.)
            score_match = re.search(r"(?i)(?:final\s+)?score\s*[:=]\s*([-+]?\d*\.?\d+)", output)
            score_report_msg = ""
            if score_match:
                current_score = float(score_match.group(1))
                if best_score == float('-inf'):
                    best_score = current_score
                    score_report_msg = f"📊 INITIAL SCORE ESTABLISHED: {current_score}"
                    print(score_report_msg)
                elif current_score > best_score:
                    improvement = current_score - best_score
                    best_score = current_score
                    score_report_msg = f"📈 SUCCESS! Score improved by {improvement:.2f} (New Best: {best_score:.2f})!"
                    print(score_report_msg)
                else:
                    score_report_msg = f"📉 NO IMPROVEMENT. Score {current_score:.2f} (Best remains: {best_score:.2f})."
                    print(score_report_msg)
            else:
                print("⚠️ Could not parse 'SCORE: <value>' from output.")

            # The Deacon automatically checks off Step 3 and injects the score into Step 4
            program_text = program_text.replace(first_unchecked_line, "- [x] **Step 3:** (Run Complete)")
            # Find and replace Step 4 to inject the output
            step_4_line = [line for line in program_text.split("\n") if "- [ ] **Step 4:**" in line]
            if step_4_line:
                program_text = program_text.replace(step_4_line[0], f"- [ ] **Step 4:** (Last Run Output:\n{output}\n{score_report_msg}\n) Did the score increase?")
            
            with open("program.md", "w") as f: f.write(program_text)
            continue # Skip LLM call, jump to next iteration

        # Otherwise, package the state and ask the LLM to do the next step (1, 2, or 4)
        user_prompt = f"""CURRENT `program.md`:
```markdown
{program_text}
```

CURRENT `universe.py`:
```python
{universe_text}
```"""
        
        try:
            llm_response = call_llm(user_prompt)
        except Exception as e:
            error_str = str(e)
            wait_time = 30
            # Try to parse the exact retry delay from the Google API error message
            retry_match = re.search(r"retryDelay': '(\d+)s", error_str) or re.search(r"retry in (\d+\.?\d*)s", error_str.lower())
            if retry_match:
                wait_time = int(float(retry_match.group(1))) + 5 # Add 5 seconds padding
            
            print(f"⚠️ API Error (Rate Limit/Quota). Sleeping for {wait_time}s...")
            time.sleep(wait_time)
            continue

        # Parse the new files from the LLM's response
        new_program = extract_code_block(llm_response, "markdown")
        new_universe = extract_code_block(llm_response, "python")

        if new_program:
            with open("program.md", "w") as f: f.write(new_program)
            print("📝 Updated program.md")
        else:
            print("⚠️ LLM failed to return a markdown block. Retrying...")
            
        if new_universe:
            with open("universe.py", "w") as f: f.write(new_universe)
            print("🧬 Updated universe.py DNA")

        time.sleep(15) # Increased rate limit breathing room from 5s to 15s

if __name__ == "__main__":
    # Ensure you are running this on the Kaggle TPU v5e / v3-8 accelerator!
    main_loop(max_iterations=100)
