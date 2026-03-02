import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
import optax
from typing import NamedTuple

# ====================== CONSTANTS ======================
NUM_ENVS = 1024
NUM_AGENTS = 2
TAPE_SIZE = 32
HIDDEN_DIM = 64
NUM_STEPS = 60          # Ticks per episode
LEARNING_RATE = 0.001
NUM_EPOCHS = 4000
ALPHA = 0.01            # Entropy bonus to encourage exploration

# Single-Byte Objects (ASCII)
EMPTY = ord('.')         # 46
BUTTON = ord('_')        # 95
DOOR_CLOSED = ord('d')   # 100
DOOR_OPEN = ord('-')     # 45
CORE = ord('z')          # 122
WALL = ord('#')          # 35
AGENT_BASE = ord('@')    # 64 (0=Silent '@', 1-26='A'-'Z')

# Fixed Map Positions
BUTTON_POS = 8
DOOR_POS = 20
CORE_POS = 24

# ====================== ENVIRONMENT ======================
class EnvState(NamedTuple):
    agent_pos: jnp.ndarray   # [NUM_ENVS, NUM_AGENTS]
    hunger: jnp.ndarray      # [NUM_ENVS, NUM_AGENTS]
    vocals: jnp.ndarray      # [NUM_ENVS, NUM_AGENTS]
    core_exists: jnp.ndarray # [NUM_ENVS]
    last_action: jnp.ndarray # [NUM_ENVS, NUM_AGENTS]
    tape: jnp.ndarray        # [NUM_ENVS, TAPE_SIZE]

def get_tape_single(pos, vocals, core_exists):
    """ Renders the 1D tape for a single universe. """
    tape = jnp.full(TAPE_SIZE, EMPTY, dtype=jnp.int32)
    tape = tape.at[BUTTON_POS].set(BUTTON)
    
    is_button_pressed = jnp.any(pos == BUTTON_POS)
    door_byte = jnp.where(is_button_pressed, DOOR_OPEN, DOOR_CLOSED)
    tape = tape.at[DOOR_POS].set(door_byte)
    
    tape = jnp.where(core_exists, tape.at[CORE_POS].set(CORE), tape)
    
    # Apply Visual Language (0 = '@', 1 = 'A', 2 = 'B', etc.)
    agent_bytes = AGENT_BASE + vocals
    tape = tape.at[pos].set(agent_bytes) 
    return tape

# Vectorize tape rendering across all 1024 environments instantly
vectorized_get_tapes = jax.vmap(get_tape_single, in_axes=(0, 0, 0))

def init_env(key: jnp.ndarray) -> EnvState:
    k1, _ = random.split(key)
    # Start agents randomly on the left side of the door
    agent_pos = random.randint(k1, (NUM_ENVS, NUM_AGENTS), 1, DOOR_POS - 2)
    hunger = jnp.ones((NUM_ENVS, NUM_AGENTS), dtype=jnp.float32)
    vocals = jnp.zeros((NUM_ENVS, NUM_AGENTS), dtype=jnp.int32)
    core_exists = jnp.ones((NUM_ENVS,), dtype=jnp.bool_)
    last_action = jnp.ones((NUM_ENVS, NUM_AGENTS), dtype=jnp.int32)
    
    tape = vectorized_get_tapes(agent_pos, vocals, core_exists)
    return EnvState(agent_pos, hunger, vocals, core_exists, last_action, tape)

def get_vision_single(pos, tape):
    """ Agent sees 4 left, 4 right. Center is masked out. """
    def get_agent_vis(p):
        offsets = jnp.arange(-4, 5)
        indices = p + offsets
        vis = jnp.where((indices >= 0) & (indices < TAPE_SIZE),
                        tape[jnp.clip(indices, 0, TAPE_SIZE-1)],
                        WALL)
        return vis.at[4].set(EMPTY) # Blind self
    return jax.vmap(get_agent_vis)(pos)

vectorized_get_vision = jax.vmap(get_vision_single, in_axes=(0, 0))

def env_step_single(pos, hunger, core_exists, actions, vocals):
    """ Physics engine for a single universe. """
    moves = actions - 1 # 0=Left, 1=Stay, 2=Right
    intended_pos = jnp.clip(pos + moves, 0, TAPE_SIZE - 1)
    
    # Stag Hunt: Door logic
    is_button_pressed = jnp.any(pos == BUTTON_POS)
    blocked_by_door = (~is_button_pressed) & (intended_pos >= DOOR_POS) & (pos < DOOR_POS)
    new_pos = jnp.where(blocked_by_door, pos, intended_pos)
    
    # Core Consumption
    at_core = (new_pos == CORE_POS) & core_exists
    anyone_ate_core = jnp.any(at_core)
    new_core_exists = jnp.where(anyone_ate_core, False, core_exists)
    
    # Metabolic Tax & Hunger
    metabolic_tax = jnp.where(vocals > 0, 0.01, 0.0) # Tax for speaking
    new_hunger = jnp.clip(hunger + 0.01 + metabolic_tax, 0.0, 1.0)
    
    # Shared Reward: If anyone eats the core, BOTH agents reset hunger to 0.0.
    new_hunger = jnp.where(anyone_ate_core, 0.0, new_hunger)
    
    new_tape = get_tape_single(new_pos, vocals, new_core_exists)
    return new_pos, new_hunger, new_core_exists, new_tape

vectorized_env_step = jax.vmap(env_step_single, in_axes=(0, 0, 0, 0, 0))

# ====================== AGENT BRAIN ======================
class AgentBrain(nn.Module):
    hidden_dim: int = HIDDEN_DIM
    
    @nn.compact
    def __call__(self, vision: jnp.ndarray, proprio: jnp.ndarray, carry):
        batch_size, num_agents, _ = vision.shape
        
        # Flatten batches to process all agents simultaneously
        v_flat = vision.reshape(-1, 9)
        p_flat = proprio.reshape(-1, 3)
        c_flat, h_flat = carry
        
        x = nn.Embed(num_embeddings=256, features=self.hidden_dim)(v_flat)
        x = x.reshape(x.shape[0], -1) 
        x = jnp.concatenate([x, p_flat], axis=-1)
        
        new_carry, lstm_out = nn.LSTMCell(features=self.hidden_dim)((c_flat, h_flat), x)
        
        # Dual Heads: Movement (3) and Vocal (27)
        policy_logits = nn.Dense(3)(lstm_out)
        vocal_logits = nn.Dense(27)(lstm_out)
        
        # Reshape back to [BATCH, NUM_AGENTS, DIMS]
        policy_logits = policy_logits.reshape(batch_size, num_agents, 3)
        vocal_logits = vocal_logits.reshape(batch_size, num_agents, 27)
        
        return policy_logits, vocal_logits, new_carry

# ====================== TRAINING LOOP ======================
model = AgentBrain()

def compute_fe(hunger):
    return jnp.square(hunger)

def rollout(params, state, key, dummy_carry, greedy=False):
    def step(carry_tuple, _):
        state, key, lstm_c = carry_tuple
        vision = vectorized_get_vision(state.agent_pos, state.tape)
        proprio = jnp.stack([state.hunger, state.last_action.astype(jnp.float32), state.vocals.astype(jnp.float32)], axis=-1)
        
        policy_logits, vocal_logits, new_lstm_c = model.apply(params, vision, proprio, lstm_c)
        
        if greedy:
            actions = jnp.argmax(policy_logits, axis=-1)
            vocals = jnp.argmax(vocal_logits, axis=-1)
            entropy = jnp.float32(0.0)
        else:
            key, k1, k2 = random.split(key, 3)
            actions = random.categorical(k1, policy_logits)
            vocals = random.categorical(k2, vocal_logits)
            
            p_probs = jax.nn.softmax(policy_logits)
            v_probs = jax.nn.softmax(vocal_logits)
            ent_p = -jnp.sum(p_probs * jnp.log(p_probs + 1e-8), axis=-1)
            ent_v = -jnp.sum(v_probs * jnp.log(v_probs + 1e-8), axis=-1)
            entropy = ent_p + ent_v
        
        new_pos, new_hunger, new_core, new_tape = vectorized_env_step(
            state.agent_pos, state.hunger, state.core_exists, actions, vocals
        )
        
        new_state = EnvState(new_pos, new_hunger, vocals, new_core, actions, new_tape)
        fe = compute_fe(new_hunger)
        
        output = {
            'policy_logits': policy_logits, 'vocal_logits': vocal_logits,
            'actions': actions, 'vocals': vocals, 'entropy': entropy,
            'fe': fe, 'agent_pos': new_state.agent_pos, 'hunger': new_state.hunger,
            'tape': new_state.tape
        }
        return (new_state, key, new_lstm_c), output
    
    _, outputs = jax.lax.scan(step, (state, key, dummy_carry), jnp.arange(NUM_STEPS))
    return outputs

@jax.jit
def update_step(params, opt_state, state, key, dummy_carry):
    def loss_fn(p):
        outputs = rollout(p, state, key, dummy_carry, greedy=False)
        fes = outputs['fe']
        
        # ACTIVE INFERENCE: The environment generates shared Free Energy.
        # Both agents want to minimize the mean societal starvation.
        shared_fe = jnp.mean(fes, axis=-1, keepdims=True) 
        rewards = -shared_fe 
        
        returns = jnp.cumsum(rewards[::-1], axis=0)[::-1]
        returns = returns - jnp.mean(returns, axis=0, keepdims=True)
        
        actions = outputs['actions']
        vocals = outputs['vocals']
        
        p_log_probs = jax.nn.log_softmax(outputs['policy_logits'])
        v_log_probs = jax.nn.log_softmax(outputs['vocal_logits'])
        
        chosen_p = jnp.take_along_axis(p_log_probs, actions[..., None], axis=-1).squeeze(-1)
        chosen_v = jnp.take_along_axis(v_log_probs, vocals[..., None], axis=-1).squeeze(-1)
        
        total_log_probs = chosen_p + chosen_v
        policy_loss = -jnp.mean(total_log_probs * returns)
        entropy_bonus = -ALPHA * jnp.mean(outputs['entropy'])
        
        return policy_loss + entropy_bonus
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss

# ====================== DEBUG UTILS ======================
def render_tape_ascii(tape_array):
    chars = []
    for b in tape_array:
        if b == EMPTY: chars.append('.')
        elif b == BUTTON: chars.append('_')
        elif b == DOOR_CLOSED: chars.append('d')
        elif b == DOOR_OPEN: chars.append('-')
        elif b == CORE: chars.append('z')
        elif b == WALL: chars.append('#')
        elif 64 <= b <= 90: chars.append(chr(b))
        else: chars.append('?')
    return '[' + "".join(chars) + ']'

def print_debug(outputs, num_print=60):
    print(f"{'Tick':<4} | {'A0 (Pos/Voc/Hunger)':<20} | {'A1 (Pos/Voc/Hunger)':<20} | {'Tape'}")
    print('-' * 90)
    for t in range(min(NUM_STEPS, num_print)):
        p0, p1 = outputs['agent_pos'][t][0]
        v0, v1 = outputs['vocals'][t][0]
        h0, h1 = outputs['hunger'][t][0]
        
        c0 = chr(AGENT_BASE + v0) if v0 > 0 else '@'
        c1 = chr(AGENT_BASE + v1) if v1 > 0 else '@'
        
        a0_str = f"{p0:02d} / {c0} / {h0:.2f}"
        a1_str = f"{p1:02d} / {c1} / {h1:.2f}"
        tape_str = render_tape_ascii(outputs['tape'][t][0])
        
        print(f"{t+1:<4} | {a0_str:<20} | {a1_str:<20} | {tape_str}")

# ====================== MAIN ======================
if __name__ == '__main__':
    key = random.PRNGKey(42)
    
    # FIX: Manually define the exact tuple structure to bypass Flax's shape inference quirks
    num_instances = NUM_ENVS * NUM_AGENTS
    dummy_carry = (
        jnp.zeros((num_instances, HIDDEN_DIM), dtype=jnp.float32), 
        jnp.zeros((num_instances, HIDDEN_DIM), dtype=jnp.float32)
    )
    
    dummy_vision = jnp.zeros((NUM_ENVS, NUM_AGENTS, 9), dtype=jnp.int32)
    dummy_proprio = jnp.zeros((NUM_ENVS, NUM_AGENTS, 3))
    
    key, init_key = random.split(key)
    params = model.init(init_key, dummy_vision, dummy_proprio, dummy_carry)
    
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(params)
    
    print(f"🚀 Training Stag Hunt & Visual Language on GPU (Envs: {NUM_ENVS}, Epochs: {NUM_EPOCHS})...")
    for epoch in range(NUM_EPOCHS):
        key, env_key, step_key = random.split(key, 3)
        state = init_env(env_key)
        
        params, opt_state, loss = update_step(params, opt_state, state, step_key, dummy_carry)
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d} | Loss {float(loss):.4f}")
    
    print("\n🎉 Training complete! Greedy test rollout for Universe 0:")
    key, test_key = random.split(key)
    test_state = init_env(test_key)
    
    test_outputs = rollout(params, test_state, test_key, dummy_carry, greedy=True)
    print_debug(test_outputs)
    
    print("\n>>> Phase 2 SUCCESS if one agent stands on '_' (flashing a letter) and the other walks through '-' to eat 'z'!")
