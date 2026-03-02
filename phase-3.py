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
NUM_STEPS = 60          
LEARNING_RATE = 0.001
NUM_EPOCHS = 4000
ALPHA = 0.01            

# Single-Byte Objects (ASCII)
EMPTY = ord('.')         
BUTTON = ord('_')        
DOOR_CLOSED = ord('d')   
DOOR_OPEN = ord('-')     
CORE = ord('z')          
WALL = ord('#')          
AGENT_BASE = ord('@')    

BUTTON_POS = 8
DOOR_POS = 20
CORE_POS = 24

# ====================== ENVIRONMENT ======================
class EnvState(NamedTuple):
    agent_pos: jnp.ndarray   
    hunger: jnp.ndarray      
    vocals: jnp.ndarray      
    core_exists: jnp.ndarray 
    last_action: jnp.ndarray 
    tape: jnp.ndarray        

def get_tape_single(pos, vocals, core_exists):
    tape = jnp.full(TAPE_SIZE, EMPTY, dtype=jnp.int32)
    tape = tape.at[BUTTON_POS].set(BUTTON)
    
    is_button_pressed = jnp.any(pos == BUTTON_POS)
    door_byte = jnp.where(is_button_pressed, DOOR_OPEN, DOOR_CLOSED)
    tape = tape.at[DOOR_POS].set(door_byte)
    
    tape = jnp.where(core_exists, tape.at[CORE_POS].set(CORE), tape)
    
    agent_bytes = AGENT_BASE + vocals
    tape = tape.at[pos].set(agent_bytes) 
    return tape

vectorized_get_tapes = jax.vmap(get_tape_single, in_axes=(0, 0, 0))

def init_env(key: jnp.ndarray) -> EnvState:
    k1, _ = random.split(key)
    agent_pos = random.randint(k1, (NUM_ENVS, NUM_AGENTS), 1, DOOR_POS - 2)
    hunger = jnp.ones((NUM_ENVS, NUM_AGENTS), dtype=jnp.float32)
    vocals = jnp.zeros((NUM_ENVS, NUM_AGENTS), dtype=jnp.int32)
    core_exists = jnp.ones((NUM_ENVS,), dtype=jnp.bool_)
    last_action = jnp.ones((NUM_ENVS, NUM_AGENTS), dtype=jnp.int32)
    
    tape = vectorized_get_tapes(agent_pos, vocals, core_exists)
    return EnvState(agent_pos, hunger, vocals, core_exists, last_action, tape)

def get_vision_single(pos, tape):
    def get_agent_vis(p):
        offsets = jnp.arange(-4, 5)
        indices = p + offsets
        vis = jnp.where((indices >= 0) & (indices < TAPE_SIZE),
                        tape[jnp.clip(indices, 0, TAPE_SIZE-1)],
                        WALL)
        return vis.at[4].set(EMPTY) 
    return jax.vmap(get_agent_vis)(pos)

vectorized_get_vision = jax.vmap(get_vision_single, in_axes=(0, 0))

def env_step_single(pos, hunger, core_exists, actions, vocals):
    moves = actions - 1 
    intended_pos = jnp.clip(pos + moves, 0, TAPE_SIZE - 1)
    
    is_button_pressed = jnp.any(pos == BUTTON_POS)
    blocked_by_door = (~is_button_pressed) & (intended_pos >= DOOR_POS) & (pos < DOOR_POS)
    new_pos = jnp.where(blocked_by_door, pos, intended_pos)
    
    at_core = (new_pos == CORE_POS) & core_exists
    anyone_ate_core = jnp.any(at_core)
    new_core_exists = jnp.where(anyone_ate_core, False, core_exists)
    
    metabolic_tax = jnp.where(vocals > 0, 0.01, 0.0) 
    new_hunger = jnp.clip(hunger + 0.01 + metabolic_tax, 0.0, 1.0)
    new_hunger = jnp.where(anyone_ate_core, 0.0, new_hunger)
    
    new_tape = get_tape_single(new_pos, vocals, new_core_exists)
    return new_pos, new_hunger, new_core_exists, new_tape

vectorized_env_step = jax.vmap(env_step_single, in_axes=(0, 0, 0, 0, 0))

# ====================== ACTIVE INFERENCE BRAIN ======================
class AgentBrain(nn.Module):
    hidden_dim: int = HIDDEN_DIM
    
    def setup(self):
        self.embed_vis = nn.Embed(num_embeddings=256, features=16)
        self.lstm = nn.LSTMCell(features=self.hidden_dim)
        
        self.policy_act = nn.Dense(3)
        self.policy_voc = nn.Dense(27)
        
        self.embed_act = nn.Embed(num_embeddings=3, features=8)
        self.embed_voc = nn.Embed(num_embeddings=27, features=16)
        self.trans_dense = nn.Dense(128)
        self.trans_out = nn.Dense(9 * 256) 

    def __call__(self, vision, proprio, carry):
        batch_size, num_agents, _ = vision.shape
        v_emb = self.embed_vis(vision)
        v_flat = v_emb.reshape(batch_size, num_agents, -1) 
        x = jnp.concatenate([v_flat, proprio], axis=-1)
        
        x_flat = x.reshape(-1, x.shape[-1])
        new_carry, lstm_out = self.lstm(carry, x_flat)
        
        p_logits = self.policy_act(lstm_out).reshape(batch_size, num_agents, 3)
        v_logits = self.policy_voc(lstm_out).reshape(batch_size, num_agents, 27)
        
        return p_logits, v_logits, new_carry, lstm_out.reshape(batch_size, num_agents, self.hidden_dim)

    def predict_next(self, h, action, vocal):
        a_emb = self.embed_act(action)
        v_emb = self.embed_voc(vocal)
        
        x = jnp.concatenate([h, a_emb, v_emb], axis=-1)
        x = nn.relu(self.trans_dense(x))
        logits = self.trans_out(x)
        
        return logits.reshape(*action.shape, 9, 256)

    def init_everything(self, vision, proprio, carry, h, action, vocal):
        _ = self.__call__(vision, proprio, carry)
        _ = self.predict_next(h, action, vocal)
        return True

# ====================== TRAINING LOOP ======================
model = AgentBrain()

def rollout(params, state, key, dummy_carry, beta, greedy=False):
    def step(carry_tuple, _):
        state, key, lstm_c = carry_tuple
        vision = vectorized_get_vision(state.agent_pos, state.tape)
        proprio = jnp.stack([state.hunger, state.last_action.astype(jnp.float32), state.vocals.astype(jnp.float32)], axis=-1)
        
        p_logits, v_logits, new_lstm_c, h = model.apply(params, vision, proprio, lstm_c, method=AgentBrain.__call__)
        
        if greedy:
            actions = jnp.argmax(p_logits, axis=-1)
            vocals = jnp.argmax(v_logits, axis=-1)
            entropy = jnp.float32(0.0)
        else:
            key, k1, k2 = random.split(key, 3)
            actions = random.categorical(k1, p_logits)
            vocals = random.categorical(k2, v_logits)
            
            p_probs, v_probs = jax.nn.softmax(p_logits), jax.nn.softmax(v_logits)
            ent_p = -jnp.sum(p_probs * jnp.log(p_probs + 1e-8), axis=-1)
            ent_v = -jnp.sum(v_probs * jnp.log(v_probs + 1e-8), axis=-1)
            entropy = ent_p + ent_v
        
        h_sg = jax.lax.stop_gradient(h)
        pred_vision_logits = model.apply(params, h_sg, actions, vocals, method=AgentBrain.predict_next)
        
        new_pos, new_hunger, new_core, new_tape = vectorized_env_step(
            state.agent_pos, state.hunger, state.core_exists, actions, vocals
        )
        new_state = EnvState(new_pos, new_hunger, vocals, new_core, actions, new_tape)
        
        new_vision = vectorized_get_vision(new_state.agent_pos, new_state.tape)
        
        # FIXED: Calculate Cross Entropy but CLIP IT so they can't farm infinite noise
        ce_loss = optax.softmax_cross_entropy_with_integer_labels(pred_vision_logits, new_vision)
        epistemic = jnp.clip(jnp.mean(ce_loss, axis=-1), 0.0, 2.0) 
        
        extrinsic = jnp.square(new_hunger)
        
        output = {
            'p_logits': p_logits, 'v_logits': v_logits,
            'actions': actions, 'vocals': vocals, 'entropy': entropy,
            'extrinsic': extrinsic, 'epistemic': epistemic,
            'shared_extrinsic': jnp.mean(extrinsic, axis=-1, keepdims=True),
            'shared_epistemic': jnp.mean(epistemic, axis=-1, keepdims=True),
            'agent_pos': new_state.agent_pos, 'hunger': new_state.hunger, 'tape': new_state.tape
        }
        return (new_state, key, new_lstm_c), output
    
    _, outputs = jax.lax.scan(step, (state, key, dummy_carry), jnp.arange(NUM_STEPS))
    return outputs

@jax.jit
def update_step(params, opt_state, state, key, dummy_carry, epoch):
    # FIXED: Start Beta at 0.1 so Survival always outweighs Curiosity
    beta = jnp.maximum(0.001, 0.1 - (epoch / 20000.0))

    def loss_fn(p):
        outputs = rollout(p, state, key, dummy_carry, beta, greedy=False)
        
        rewards = -outputs['shared_extrinsic'] + (beta * outputs['shared_epistemic'])
        
        returns = jnp.cumsum(rewards[::-1], axis=0)[::-1]
        returns = returns - jnp.mean(returns, axis=0, keepdims=True)
        
        actions, vocals = outputs['actions'], outputs['vocals']
        p_log_probs = jax.nn.log_softmax(outputs['p_logits'])
        v_log_probs = jax.nn.log_softmax(outputs['v_logits'])
        
        chosen_p = jnp.take_along_axis(p_log_probs, actions[..., None], axis=-1).squeeze(-1)
        chosen_v = jnp.take_along_axis(v_log_probs, vocals[..., None], axis=-1).squeeze(-1)
        
        policy_loss = -jnp.mean((chosen_p + chosen_v) * returns)
        
        transition_loss = jnp.mean(outputs['epistemic'])
        entropy_bonus = -ALPHA * jnp.mean(outputs['entropy'])
        
        total_loss = policy_loss + transition_loss + entropy_bonus
        return total_loss, (jnp.mean(outputs['extrinsic']), jnp.mean(outputs['epistemic']), beta)
    
    (loss, (avg_ext, avg_epi, cur_beta)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss, avg_ext, avg_epi, cur_beta

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
    
    num_instances = NUM_ENVS * NUM_AGENTS
    dummy_carry = (
        jnp.zeros((num_instances, HIDDEN_DIM), dtype=jnp.float32), 
        jnp.zeros((num_instances, HIDDEN_DIM), dtype=jnp.float32)
    )
    dummy_vision = jnp.zeros((NUM_ENVS, NUM_AGENTS, 9), dtype=jnp.int32)
    dummy_proprio = jnp.zeros((NUM_ENVS, NUM_AGENTS, 3))
    dummy_h = jnp.zeros((NUM_ENVS, NUM_AGENTS, HIDDEN_DIM), dtype=jnp.float32)
    dummy_act = jnp.zeros((NUM_ENVS, NUM_AGENTS), dtype=jnp.int32)
    dummy_voc = jnp.zeros((NUM_ENVS, NUM_AGENTS), dtype=jnp.int32)
    
    key, init_key = random.split(key)
    params = model.init(
        init_key, dummy_vision, dummy_proprio, dummy_carry, 
        dummy_h, dummy_act, dummy_voc, 
        method=AgentBrain.init_everything
    )
    
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(params)
    
    print(f"🚀 Training Phase 3: Deep Active Inference (Envs: {NUM_ENVS}, Epochs: {NUM_EPOCHS})...")
    for epoch in range(NUM_EPOCHS):
        key, env_key, step_key = random.split(key, 3)
        state = init_env(env_key)
        
        params, opt_state, loss, avg_ext, avg_epi, beta = update_step(params, opt_state, state, step_key, dummy_carry, epoch)
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d} | Loss: {float(loss):.4f} | Extrinsic (Pragmatic): {float(avg_ext):.4f} | Epistemic (Curiosity): {float(avg_epi):.4f} | Beta: {float(beta):.3f}")
    
    print("\n🎉 Training complete! Greedy test rollout for Universe 0:")
    key, test_key = random.split(key)
    test_state = init_env(test_key)
    
    test_outputs = rollout(params, test_state, test_key, dummy_carry, beta=0.0, greedy=True)
    print_debug(test_outputs)
