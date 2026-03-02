import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
import optax
from typing import NamedTuple

# ====================== CONSTANTS ======================
EMPTY = 0
AGENT = 1   # '@'
BERRY = 2   # 'B'
WALL = 255  # '#'
TAPE_SIZE = 16
HIDDEN_DIM = 64
NUM_STEPS = 100
ALPHA = 0.01          # Entropy bonus coefficient
LEARNING_RATE = 0.001
NUM_EPOCHS = 3000     

class EnvState(NamedTuple):
    tape: jnp.ndarray
    agent_pos: jnp.int32
    berry_pos: jnp.int32
    hunger: jnp.float32
    last_action: jnp.int32

def init_env(key: jnp.ndarray) -> EnvState:
    k1, k2 = random.split(key, 2)
    agent_pos = random.randint(k1, (), 0, TAPE_SIZE)
    berry_pos = random.randint(k2, (), 0, TAPE_SIZE)
    # Ensure berry doesn't spawn exactly on the agent
    berry_pos = jnp.where(berry_pos == agent_pos, (berry_pos + 1) % TAPE_SIZE, berry_pos)
    
    tape = jnp.zeros(TAPE_SIZE, dtype=jnp.uint8)
    tape = tape.at[berry_pos].set(BERRY)
    tape = tape.at[agent_pos].set(AGENT)
    
    return EnvState(tape, agent_pos, berry_pos, jnp.float32(1.0), jnp.int32(-1))

@jax.jit
def get_vision(state: EnvState) -> jnp.ndarray:
    # Agent sees 4 left, 4 right. Center (4) is masked out.
    offsets = jnp.arange(-4, 5)
    indices = state.agent_pos + offsets
    vision = jnp.where(
        (indices >= 0) & (indices < TAPE_SIZE),
        state.tape[jnp.clip(indices, 0, TAPE_SIZE-1)],
        WALL
    )
    # Mask out the agent's own body
    vision = vision.at[4].set(EMPTY)
    return vision.astype(jnp.int32) # Cast to int32 for the Embedding layer

@jax.jit
def env_step(state: EnvState, action: jnp.int32, key: jnp.ndarray) -> EnvState:
    # Actions: 0=Left, 1=Stay, 2=Right
    move = action - 1
    new_pos = jnp.clip(state.agent_pos + move, 0, TAPE_SIZE - 1)
    
    # Hunger mechanics (increases slowly, resets when berry eaten)
    hunger = jnp.clip(state.hunger + 0.01, 0.0, 1.0)
    eaten = (new_pos == state.berry_pos)
    hunger = jnp.where(eaten, 0.0, hunger)
    
    # Respawn berry if eaten
    k1, _ = random.split(key)
    new_berry_pos = jnp.where(eaten, random.randint(k1, (), 0, TAPE_SIZE), state.berry_pos)
    new_berry_pos = jnp.where(eaten & (new_berry_pos == new_pos), (new_berry_pos + 1) % TAPE_SIZE, new_berry_pos)
    
    # Rebuild the 1D tape
    new_tape = jnp.zeros(TAPE_SIZE, dtype=jnp.uint8)
    new_tape = new_tape.at[new_berry_pos].set(BERRY)
    new_tape = new_tape.at[new_pos].set(AGENT)
    
    return EnvState(new_tape, new_pos, new_berry_pos, hunger, action)

class AgentBrain(nn.Module):
    hidden_dim: int = HIDDEN_DIM
    
    @nn.compact
    def __call__(self, vision: jnp.ndarray, proprio: jnp.ndarray, carry):
        # 1. Embed the 9-tile vision (256 possible byte values)
        x = nn.Embed(num_embeddings=256, features=self.hidden_dim)(vision)
        x = x.reshape(-1) # Flatten
        
        # 2. Append proprioception (hunger, last action)
        x = jnp.concatenate([x, proprio])
        
        # 3. LSTM Memory Core
        new_carry, x = nn.LSTMCell(features=self.hidden_dim)(carry, x)
        
        # 4. Action Logits
        policy_logits = nn.Dense(3)(x)
        return policy_logits, new_carry

def compute_fe(hunger: jnp.float32) -> jnp.float32:
    # Free Energy is the divergence from the biological prior (hunger should be 0)
    return jnp.square(hunger)

# Instantiate the model globally so JAX can use it in pure functions
model = AgentBrain()

def rollout(params, state, key, lstm_carry, greedy=False):
    """ Runs NUM_STEPS of the environment and collects data. """
    def step(carry_tuple, _):
        state, key, lstm_c = carry_tuple
        vision = get_vision(state)
        proprio = jnp.array([state.hunger, jnp.float32(state.last_action), 0.0])
        
        # Forward pass
        policy_logits, new_lstm_c = model.apply(params, vision, proprio, lstm_c)
        
        if greedy:
            action = jnp.argmax(policy_logits)
            entropy = jnp.float32(0.0)
        else:
            key, subkey = random.split(key)
            action = random.categorical(subkey, policy_logits)
            probs = jax.nn.softmax(policy_logits)
            entropy = -jnp.sum(probs * jnp.log(probs + 1e-8))
        
        # Step environment
        key, subkey = random.split(key)
        new_state = env_step(state, action, subkey)
        fe = compute_fe(new_state.hunger)
        
        output = {
            'logits': policy_logits,
            'action': action,
            'entropy': entropy,
            'fe': fe,
            'agent_pos': new_state.agent_pos,
            'hunger': new_state.hunger,
            'tape': new_state.tape
        }
        return (new_state, key, new_lstm_c), output
    
    _, outputs = jax.lax.scan(step, (state, key, lstm_carry), jnp.arange(NUM_STEPS))
    return outputs

@jax.jit
def update_step(params, opt_state, state, key, dummy_carry):
    """ JIT-compiled training step. Runs the rollout and updates weights. """
    def loss_fn(p):
        outputs = rollout(p, state, key, dummy_carry, greedy=False)
        fes = outputs['fe']
        
        # In Phase 1, we use Policy Gradient to minimize future Free Energy (hunger)
        rewards = -fes 
        returns = jnp.cumsum(rewards[::-1])[::-1]
        returns = returns - jnp.mean(returns) # Baseline stabilization
        
        actions = outputs['action']
        logits = outputs['logits']
        log_probs = jax.nn.log_softmax(logits)[jnp.arange(NUM_STEPS), actions]
        
        policy_loss = -jnp.mean(log_probs * returns)
        entropy_bonus = -ALPHA * jnp.mean(outputs['entropy'])
        
        loss = policy_loss + entropy_bonus
        return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, opt_state, loss

def render_tape(tape):
    # Renders the single-byte integers into ASCII for human viewing
    char_map = {EMPTY: '.', AGENT: '@', BERRY: 'B', WALL: '#'}
    return '[' + ''.join(char_map.get(int(b), '?') for b in tape) + ']'

def print_debug(outputs, num_print=25):
    print(f"{'Tick':<5} | {'Pos':<4} | {'Hunger':<6} | {'FE':<8} | Tape")
    print('-' * 65)
    for t in range(min(NUM_STEPS, num_print)):
        print(f"{t+1:<5} | {int(outputs['agent_pos'][t]):<4} | "
              f"{float(outputs['hunger'][t]):<6.2f} | "
              f"{float(outputs['fe'][t]):<8.4f} | "
              f"{render_tape(outputs['tape'][t])}")

# ====================== MAIN ======================
if __name__ == '__main__':
    key = random.PRNGKey(42)
    
    # Initialize parameters
    dummy_vision = jnp.zeros(9, dtype=jnp.int32)
    dummy_proprio = jnp.zeros(3)
    dummy_carry = (jnp.zeros((HIDDEN_DIM,)), jnp.zeros((HIDDEN_DIM,))) # Clean LSTM init
    
    key, init_key = random.split(key)
    params = model.init(init_key, dummy_vision, dummy_proprio, dummy_carry)
    
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(params)
    
    print("🚀 Training Berry Hunter on GPU (3000 epochs)...")
    for epoch in range(NUM_EPOCHS):
        key, env_key, step_key = random.split(key, 3)
        state = init_env(env_key)
        
        # The entire forward pass, environment simulation, and backward pass happens here
        params, opt_state, loss = update_step(params, opt_state, state, step_key, dummy_carry)
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d} | Loss {float(loss):.4f}")
    
    print("\n🎉 Training complete! Greedy test rollout:")
    key, test_key = random.split(key)
    test_state = init_env(test_key)
    
    # Run a test rollout using deterministic (greedy) actions
    test_outputs = rollout(params, test_state, test_key, dummy_carry, greedy=True)
    print_debug(test_outputs)
    
    print("\n>>> Phase 1 SUCCESS if agent reliably finds & eats berries (hunger -> 0, FE near 0)!")
