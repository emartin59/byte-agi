import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
import optax
from typing import NamedTuple
import numpy as np

# ====================== CONSTANTS ======================
NUM_ENVS = 1024
NUM_AGENTS = 2
TAPE_SIZE = 32
HIDDEN_DIM = 64
NUM_STEPS = 80          # Increased to allow signal propagation time
LEARNING_RATE = 0.001
NUM_EPOCHS = 5000
ALPHA = 0.01            

# Single-Byte Objects (ASCII)
EMPTY = ord('.')         
WALL = ord('#')          
BUTTON = ord('_')        
BUTTON_ON = ord('=')     
WIRE = ord('w')
WIRE_ON = ord('*')
DOOR_CLOSED = ord('d')   
DOOR_OPEN = ord('-')     
CORE = ord('z')          
AGENT_BASE = ord('@')    

# Map Layout (Agents spawn at 1-3)
BUTTON_POS = 4
WIRE_START = 5
WIRE_END = 12
DOOR_POS = 13
CORE_POS = 14

# ====================== ENVIRONMENT & CELLULAR AUTOMATA ======================
class EnvState(NamedTuple):
    agent_pos: jnp.ndarray   
    hunger: jnp.ndarray      
    vocals: jnp.ndarray      
    last_action: jnp.ndarray 
    tape: jnp.ndarray        

def apply_ca_physics(tape, agent_pos, vocals):
    """ 
    Rule-110 Style 1D Cellular Automaton.
    Physics emerge from purely local 3-byte interactions (Left, Center, Agent).
    """
    idx = jnp.arange(TAPE_SIZE)
    a_pos_expand = agent_pos[:, None] 
    speaking_expand = (vocals > 0)[:, None] 
    
    # Check if an actively speaking agent is standing on a tile
    is_agent_speaking_here = jnp.any((a_pos_expand == idx) & speaking_expand, axis=0)
    
    L = jnp.roll(tape, 1) # Look at the left neighbor
    
    new_tape = tape
    
    # 1. Button Physics (Agent must stand on it AND speak to activate it)
    new_tape = jnp.where((tape == BUTTON) & is_agent_speaking_here, BUTTON_ON, new_tape)
    new_tape = jnp.where((tape == BUTTON_ON) & ~is_agent_speaking_here, BUTTON, new_tape)
    
    # 2. Wire Physics (Signal propagates left-to-right, 1 tile per tick)
    signal_incoming = (L == BUTTON_ON) | (L == WIRE_ON)
    new_tape = jnp.where((tape == WIRE) & signal_incoming, WIRE_ON, new_tape)
    new_tape = jnp.where((tape == WIRE_ON) & ~signal_incoming, WIRE, new_tape)
    
    # 3. Door Physics (Opens if powered by a wire or button on its left)
    new_tape = jnp.where((tape == DOOR_CLOSED) & signal_incoming, DOOR_OPEN, new_tape)
    new_tape = jnp.where((tape == DOOR_OPEN) & ~signal_incoming, DOOR_CLOSED, new_tape)
    
    return new_tape

def build_base_tape():
    tape = jnp.full(TAPE_SIZE, EMPTY, dtype=jnp.int32)
    tape = tape.at[0].set(WALL)
    tape = tape.at[-1].set(WALL)
    tape = tape.at[BUTTON_POS].set(BUTTON)
    tape = tape.at[WIRE_START:WIRE_END+1].set(WIRE)
    tape = tape.at[DOOR_POS].set(DOOR_CLOSED)
    tape = tape.at[CORE_POS].set(CORE)
    return tape

def init_env(key: jnp.ndarray) -> EnvState:
    k1, _ = random.split(key)
    agent_pos = random.randint(k1, (NUM_ENVS, NUM_AGENTS), 1, BUTTON_POS)
    hunger = jnp.ones((NUM_ENVS, NUM_AGENTS), dtype=jnp.float32)
    vocals = jnp.zeros((NUM_ENVS, NUM_AGENTS), dtype=jnp.int32)
    last_action = jnp.ones((NUM_ENVS, NUM_AGENTS), dtype=jnp.int32)
    
    base_tapes = jnp.tile(build_base_tape(), (NUM_ENVS, 1))
    return EnvState(agent_pos, hunger, vocals, last_action, base_tapes)

def render_agents_on_tape(tape, pos, vocals):
    agent_bytes = AGENT_BASE + vocals
    return tape.at[pos].set(agent_bytes)

def get_vision_single(pos, tape_with_agents):
    def get_agent_vis(p):
        offsets = jnp.arange(-4, 5)
        indices = p + offsets
        vis = jnp.where((indices >= 0) & (indices < TAPE_SIZE),
                        tape_with_agents[jnp.clip(indices, 0, TAPE_SIZE-1)],
                        WALL)
        return vis.at[4].set(EMPTY) # Blind self
    return jax.vmap(get_agent_vis)(pos)

vectorized_get_vision = jax.vmap(get_vision_single, in_axes=(0, 0))
vectorized_render_agents = jax.vmap(render_agents_on_tape, in_axes=(0, 0, 0))
vectorized_ca_physics = jax.vmap(apply_ca_physics, in_axes=(0, 0, 0))

def env_step_single(pos, hunger, tape, actions, vocals):
    moves = actions - 1 
    intended_pos = jnp.clip(pos + moves, 0, TAPE_SIZE - 1)
    
    # Door Collision (Door is solid if closed)
    door_is_closed = (tape[DOOR_POS] == DOOR_CLOSED)
    blocked_by_door = door_is_closed & (intended_pos >= DOOR_POS) & (pos < DOOR_POS)
    new_pos = jnp.where(blocked_by_door, pos, intended_pos)
    
    # Run Cellular Automata Physics
    new_tape = apply_ca_physics(tape, new_pos, vocals)
    
    # Core Consumption
    at_core = (new_pos == CORE_POS) & (new_tape[CORE_POS] == CORE)
    anyone_ate_core = jnp.any(at_core)
    new_tape = jnp.where(anyone_ate_core, new_tape.at[CORE_POS].set(EMPTY), new_tape)
    
    # Metabolic Tax
    metabolic_tax = jnp.where(vocals > 0, 0.01, 0.0) 
    new_hunger = jnp.clip(hunger + 0.01 + metabolic_tax, 0.0, 1.0)
    new_hunger = jnp.where(anyone_ate_core, 0.0, new_hunger)
    
    return new_pos, new_hunger, new_tape

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
        return self.trans_out(x).reshape(*action.shape, 9, 256)

    def init_everything(self, vision, proprio, carry, h, action, vocal):
        _ = self.__call__(vision, proprio, carry)
        _ = self.predict_next(h, action, vocal)
        return True

# ====================== TRAINING & ANALYSIS ======================
model = AgentBrain()

def rollout(params, state, key, dummy_carry, beta, greedy=False):
    def step(carry_tuple, _):
        state, key, lstm_c = carry_tuple
        tape_with_agents = vectorized_render_agents(state.tape, state.agent_pos, state.vocals)
        vision = vectorized_get_vision(state.agent_pos, tape_with_agents)
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
            entropy = -jnp.sum(p_probs * jnp.log(p_probs + 1e-8), axis=-1) - jnp.sum(v_probs * jnp.log(v_probs + 1e-8), axis=-1)
        
        h_sg = jax.lax.stop_gradient(h)
        pred_vision_logits = model.apply(params, h_sg, actions, vocals, method=AgentBrain.predict_next)
        
        new_pos, new_hunger, new_tape = vectorized_env_step(state.agent_pos, state.hunger, state.tape, actions, vocals)
        new_state = EnvState(new_pos, new_hunger, vocals, actions, new_tape)
        
        new_tape_with_agents = vectorized_render_agents(new_state.tape, new_state.agent_pos, new_state.vocals)
        new_vision = vectorized_get_vision(new_state.agent_pos, new_tape_with_agents)
        
        ce_loss = optax.softmax_cross_entropy_with_integer_labels(pred_vision_logits, new_vision)
        epistemic = jnp.clip(jnp.mean(ce_loss, axis=-1), 0.0, 2.0) 
        extrinsic = jnp.square(new_hunger)
        
        output = {
            'p_logits': p_logits, 'v_logits': v_logits, 'actions': actions, 'vocals': vocals, 'entropy': entropy,
            'shared_extrinsic': jnp.mean(extrinsic, axis=-1, keepdims=True),
            'shared_epistemic': jnp.mean(epistemic, axis=-1, keepdims=True),
            'agent_pos': new_state.agent_pos, 'hunger': new_state.hunger, 'tape': new_tape_with_agents
        }
        return (new_state, key, new_lstm_c), output
    
    _, outputs = jax.lax.scan(step, (state, key, dummy_carry), jnp.arange(NUM_STEPS))
    return outputs

@jax.jit
def update_step(params, opt_state, state, key, dummy_carry, epoch):
    beta = jnp.maximum(0.001, 0.1 - (epoch / 20000.0))

    def loss_fn(p):
        outputs = rollout(p, state, key, dummy_carry, beta, greedy=False)
        rewards = -outputs['shared_extrinsic'] + (beta * outputs['shared_epistemic'])
        
        returns = jnp.cumsum(rewards[::-1], axis=0)[::-1]
        returns = returns - jnp.mean(returns, axis=0, keepdims=True)
        
        p_log_probs, v_log_probs = jax.nn.log_softmax(outputs['p_logits']), jax.nn.log_softmax(outputs['v_logits'])
        chosen_p = jnp.take_along_axis(p_log_probs, outputs['actions'][..., None], axis=-1).squeeze(-1)
        chosen_v = jnp.take_along_axis(v_log_probs, outputs['vocals'][..., None], axis=-1).squeeze(-1)
        
        policy_loss = -jnp.mean((chosen_p + chosen_v) * returns)
        transition_loss = jnp.mean(outputs['shared_epistemic'])
        entropy_bonus = -ALPHA * jnp.mean(outputs['entropy'])
        
        return policy_loss + transition_loss + entropy_bonus, outputs
    
    (loss, outputs), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss, jnp.mean(outputs['shared_extrinsic']), jnp.mean(outputs['shared_epistemic']), beta, outputs

# ====================== LANGUAGE ANALYSIS SUITE ======================
def analyze_language(outputs):
    """ Extracts vocabulary distribution and convention success rate. """
    all_vocals = np.array(outputs['vocals'])
    
    # FIXED WIN RATE: Check if the AGI Core (z) is missing from the final tape
    # If the core is gone, it means the agents successfully navigated the door and ate it.
    final_tapes = np.array(outputs['tape'][-1])
    success_rate = np.mean((final_tapes[:, CORE_POS] != CORE)) * 100
    
    spoken = all_vocals[all_vocals > 0]
    if len(spoken) == 0:
        return "Agents are completely silent.", success_rate
        
    unique, counts = np.unique(spoken, return_counts=True)
    sorted_idx = np.argsort(-counts)
    
    top_3 = []
    for i in range(min(3, len(unique))):
        char = chr(AGENT_BASE + unique[sorted_idx[i]])
        pct = (counts[sorted_idx[i]] / len(spoken)) * 100
        top_3.append(f"'{char}': {pct:.1f}%")
        
    return " | ".join(top_3), success_rate

def render_tape_ascii(tape_array):
    char_map = {EMPTY: '.', BUTTON: '_', BUTTON_ON: '=', WIRE: 'w', WIRE_ON: '*', 
                DOOR_CLOSED: 'd', DOOR_OPEN: '-', CORE: 'z', WALL: '#'}
    return '[' + "".join([char_map.get(int(b), chr(b) if 64 <= b <= 90 else '?') for b in tape_array]) + ']'

def print_debug(outputs):
    print(f"{'Tick':<4} | {'A0 (Pos/Voc/Hunger)':<20} | {'A1 (Pos/Voc/Hunger)':<20} | {'Tape'}")
    print('-' * 95)
    for t in range(min(NUM_STEPS, 40)):
        p0, p1 = outputs['agent_pos'][t][0]
        v0, v1 = outputs['vocals'][t][0]
        h0, h1 = outputs['hunger'][t][0]
        c0, c1 = (chr(AGENT_BASE + v0) if v0 > 0 else '@'), (chr(AGENT_BASE + v1) if v1 > 0 else '@')
        print(f"{t+1:<4} | {p0:02d} / {c0} / {h0:.2f}{'':<6} | {p1:02d} / {c1} / {h1:.2f}{'':<6} | {render_tape_ascii(outputs['tape'][t][0])}")

# ====================== MAIN ======================
if __name__ == '__main__':
    key = random.PRNGKey(42)
    
    num_instances = NUM_ENVS * NUM_AGENTS
    dummy_carry = (jnp.zeros((num_instances, HIDDEN_DIM)), jnp.zeros((num_instances, HIDDEN_DIM)))
    dummy_vision = jnp.zeros((NUM_ENVS, NUM_AGENTS, 9), dtype=jnp.int32)
    dummy_proprio = jnp.zeros((NUM_ENVS, NUM_AGENTS, 3))
    dummy_h = jnp.zeros((NUM_ENVS, NUM_AGENTS, HIDDEN_DIM))
    dummy_act, dummy_voc = jnp.zeros((NUM_ENVS, NUM_AGENTS), dtype=jnp.int32), jnp.zeros((NUM_ENVS, NUM_AGENTS), dtype=jnp.int32)
    
    key, init_key = random.split(key)
    params = model.init(init_key, dummy_vision, dummy_proprio, dummy_carry, dummy_h, dummy_act, dummy_voc, method=AgentBrain.init_everything)
    
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(params)
    
    print(f"🚀 Training Phase 4: CA Physics & Language Emergence (Envs: {NUM_ENVS})...")
    for epoch in range(NUM_EPOCHS):
        key, env_key, step_key = random.split(key, 3)
        state = init_env(env_key)
        
        params, opt_state, loss, avg_ext, avg_epi, beta, outputs = update_step(params, opt_state, state, step_key, dummy_carry, epoch)
        
        if epoch % 500 == 0:
            top_vocab, win_rate = analyze_language(outputs)
            print(f"Epoch {epoch:4d} | Loss: {float(loss):.4f} | Ext: {float(avg_ext):.4f} | Epi: {float(avg_epi):.4f}")
            print(f"   -> Win Rate: {win_rate:.1f}% | Top Vocab: {top_vocab}")
    
    print("\n🎉 Training complete! Greedy test rollout for Universe 0:")
    key, test_key = random.split(key)
    test_state = init_env(test_key)
    test_outputs = rollout(params, test_state, test_key, dummy_carry, beta=0.0, greedy=True)
    print_debug(test_outputs)
