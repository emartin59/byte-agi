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
NUM_STEPS = 80          
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

# ====================== UED ENVIRONMENT & PHYSICS ======================
class EnvState(NamedTuple):
    agent_pos: jnp.ndarray   
    hunger: jnp.ndarray      
    vocals: jnp.ndarray      
    last_action: jnp.ndarray 
    tape: jnp.ndarray        

def init_env_single(key: jnp.ndarray) -> EnvState:
    k1, k2, k3 = random.split(key, 3)
    
    # UED: Randomize map layout every single episode
    button_pos = random.randint(k1, (), 3, 10)
    wire_len = random.randint(k2, (), 3, 10)
    door_pos = button_pos + wire_len + 1
    core_pos = door_pos + 1
    
    tape = jnp.full(TAPE_SIZE, EMPTY, dtype=jnp.int32)
    tape = tape.at[0].set(WALL)
    tape = tape.at[-1].set(WALL)
    tape = tape.at[button_pos].set(BUTTON)
    
    idx = jnp.arange(TAPE_SIZE)
    is_wire = (idx > button_pos) & (idx < door_pos)
    tape = jnp.where(is_wire, WIRE, tape)
    
    tape = tape.at[door_pos].set(DOOR_CLOSED)
    tape = tape.at[core_pos].set(CORE)
    
    # Agents spawn safely to the left of the randomized button
    agent_pos = random.randint(k3, (NUM_AGENTS,), 1, button_pos)
    hunger = jnp.ones((NUM_AGENTS,), dtype=jnp.float32)
    vocals = jnp.zeros((NUM_AGENTS,), dtype=jnp.int32)
    last_action = jnp.ones((NUM_AGENTS,), dtype=jnp.int32)
    
    return EnvState(agent_pos, hunger, vocals, last_action, tape)

vectorized_init_env = jax.vmap(init_env_single, in_axes=(0,))

def apply_ca_physics_single(tape, agent_pos, vocals):
    """ Rule-110 Style 1D Cellular Automaton. Works anywhere on the tape. """
    idx = jnp.arange(TAPE_SIZE)
    a_pos_expand = agent_pos[:, None] 
    speaking_expand = (vocals > 0)[:, None] 
    
    is_agent_speaking_here = jnp.any((a_pos_expand == idx) & speaking_expand, axis=0)
    L = jnp.roll(tape, 1)
    
    new_tape = tape
    new_tape = jnp.where((tape == BUTTON) & is_agent_speaking_here, BUTTON_ON, new_tape)
    new_tape = jnp.where((tape == BUTTON_ON) & ~is_agent_speaking_here, BUTTON, new_tape)
    
    signal_incoming = (L == BUTTON_ON) | (L == WIRE_ON)
    new_tape = jnp.where((tape == WIRE) & signal_incoming, WIRE_ON, new_tape)
    new_tape = jnp.where((tape == WIRE_ON) & ~signal_incoming, WIRE, new_tape)
    
    new_tape = jnp.where((tape == DOOR_CLOSED) & signal_incoming, DOOR_OPEN, new_tape)
    new_tape = jnp.where((tape == DOOR_OPEN) & ~signal_incoming, DOOR_CLOSED, new_tape)
    return new_tape

def render_agents_single(tape, pos, vocals):
    agent_bytes = AGENT_BASE + vocals
    return tape.at[pos].set(agent_bytes)

def get_vision_single(pos, tape_with_agents):
    def get_agent_vis(p):
        offsets = jnp.arange(-4, 5)
        indices = p + offsets
        vis = jnp.where((indices >= 0) & (indices < TAPE_SIZE),
                        tape_with_agents[jnp.clip(indices, 0, TAPE_SIZE-1)],
                        WALL)
        return vis.at[4].set(EMPTY) 
    return jax.vmap(get_agent_vis)(pos)

def env_step_single(pos, hunger, tape, actions, vocals):
    moves = actions - 1 
    intended_pos = jnp.clip(pos + moves, 0, TAPE_SIZE - 1)
    
    # Generalized Collision: Cannot step on a Wall or a Closed Door
    blocked = (tape[intended_pos] == DOOR_CLOSED) | (tape[intended_pos] == WALL)
    new_pos = jnp.where(blocked, pos, intended_pos)
    
    new_tape = apply_ca_physics_single(tape, new_pos, vocals)
    
    # Generalized Consumption: Eat the Core wherever it is
    at_core = (new_tape[new_pos] == CORE)
    anyone_ate_core = jnp.any(at_core)
    new_tape = jnp.where(anyone_ate_core, jnp.where(new_tape == CORE, EMPTY, new_tape), new_tape)
    
    metabolic_tax = jnp.where(vocals > 0, 0.01, 0.0) 
    new_hunger = jnp.clip(hunger + 0.01 + metabolic_tax, 0.0, 1.0)
    new_hunger = jnp.where(anyone_ate_core, 0.0, new_hunger)
    
    return new_pos, new_hunger, new_tape

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
        v_emb = self.embed_vis(vision)
        v_flat = v_emb.reshape(NUM_AGENTS, -1) 
        x = jnp.concatenate([v_flat, proprio], axis=-1)
        
        new_carry, lstm_out = self.lstm(carry, x)
        
        p_logits = self.policy_act(lstm_out)
        v_logits = self.policy_voc(lstm_out)
        return p_logits, v_logits, new_carry, lstm_out

    def predict_next(self, h, action, vocal):
        a_emb = self.embed_act(action)
        v_emb = self.embed_voc(vocal)
        x = jnp.concatenate([h, a_emb, v_emb], axis=-1)
        x = nn.relu(self.trans_dense(x))
        return self.trans_out(x).reshape(NUM_AGENTS, 9, 256)

    def init_everything(self, vision, proprio, carry, h, action, vocal):
        _ = self.__call__(vision, proprio, carry)
        _ = self.predict_next(h, action, vocal)
        return True

# ====================== TRAINING & EVOLUTION LOGIC ======================
model = AgentBrain()

def rollout_single(params, state, key, dummy_carry, beta, greedy=False):
    def step(carry_tuple, _):
        state, key, lstm_c = carry_tuple
        tape_with_agents = render_agents_single(state.tape, state.agent_pos, state.vocals)
        vision = get_vision_single(state.agent_pos, tape_with_agents)
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
        
        new_pos, new_hunger, new_tape = env_step_single(state.agent_pos, state.hunger, state.tape, actions, vocals)
        new_state = EnvState(new_pos, new_hunger, vocals, actions, new_tape)
        
        new_tape_with_agents = render_agents_single(new_state.tape, new_state.agent_pos, new_state.vocals)
        new_vision = get_vision_single(new_state.agent_pos, new_tape_with_agents)
        
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

def loss_fn_single(p, state, key, dummy_carry, beta):
    outputs = rollout_single(p, state, key, dummy_carry, beta, greedy=False)
    rewards = -outputs['shared_extrinsic'] + (beta * outputs['shared_epistemic'])
    
    returns = jnp.cumsum(rewards[::-1], axis=0)[::-1]
    returns = returns - jnp.mean(returns, axis=0, keepdims=True)
    
    p_log_probs, v_log_probs = jax.nn.log_softmax(outputs['p_logits']), jax.nn.log_softmax(outputs['v_logits'])
    chosen_p = jnp.take_along_axis(p_log_probs, outputs['actions'][..., None], axis=-1).squeeze(-1)
    chosen_v = jnp.take_along_axis(v_log_probs, outputs['vocals'][..., None], axis=-1).squeeze(-1)
    
    policy_loss = -jnp.mean((chosen_p + chosen_v) * returns)
    transition_loss = jnp.mean(outputs['shared_epistemic'])
    entropy_bonus = -ALPHA * jnp.mean(outputs['entropy'])
    
    total_loss = policy_loss + transition_loss + entropy_bonus
    return total_loss, (jnp.mean(outputs['shared_extrinsic']), jnp.mean(outputs['shared_epistemic']), outputs)

@jax.jit
def update_batch(params_batch, opt_state_batch, state_batch, key_batch, dummy_carry, epoch):
    beta = jnp.maximum(0.001, 0.1 - (epoch / 20000.0))
    vg_fn = jax.vmap(jax.value_and_grad(loss_fn_single, has_aux=True), in_axes=(0, 0, 0, None, None))
    (loss_batch, (ext_batch, epi_batch, outputs_batch)), grads_batch = vg_fn(params_batch, state_batch, key_batch, dummy_carry, beta)
    
    updates, new_opt_state = optimizer.update(grads_batch, opt_state_batch, params_batch)
    new_params = optax.apply_updates(params_batch, updates)
    
    return new_params, new_opt_state, jnp.mean(loss_batch), jnp.mean(ext_batch), jnp.mean(epi_batch), beta, outputs_batch

@jax.jit
def evolve_population(params_batch, opt_state_batch, ext_batch, key):
    """ Kills the bottom 50% (high hunger). Clones top 50% with mutation. """
    sort_idx = jnp.argsort(ext_batch)
    top_idx = sort_idx[:NUM_ENVS // 2]
    
    def apply_evolution(x):
        if x.ndim == 0:  # <--- FIX: Do not slice scalar variables (like optimizer step count)
            return x
        top_half = x[top_idx]
        return jnp.concatenate([top_half, top_half], axis=0)

    new_params = jax.tree_util.tree_map(apply_evolution, params_batch)
    new_opt_state = jax.tree_util.tree_map(apply_evolution, opt_state_batch)
    
    def apply_noise(p, k):
        if p.ndim == 0:  # <--- FIX: Do not add noise to scalars
            return p
        noise = jax.random.normal(k, p.shape) * 0.05
        mask = jnp.concatenate([jnp.zeros(NUM_ENVS // 2), jnp.ones(NUM_ENVS // 2)])
        mask = mask.reshape([NUM_ENVS] + [1] * (p.ndim - 1))
        return p + (noise * mask)

    leaves, treedef = jax.tree_util.tree_flatten(new_params)
    keys = jax.random.split(key, len(leaves))
    noisy_leaves = [apply_noise(leaf, k) for leaf, k in zip(leaves, keys)]
    final_params = jax.tree_util.tree_unflatten(treedef, noisy_leaves)
    
    return final_params, new_opt_state

# ====================== ANALYSIS & DEBUG ======================
def analyze_language(outputs):
    all_vocals = np.array(outputs['vocals'])
    final_tapes = np.array(outputs['tape'][-1])
    
    # Generalized Win Rate: check if AGI Core is missing from anywhere on the final tape
    success_rate = np.mean(np.all(final_tapes != CORE, axis=-1)) * 100
    
    spoken = all_vocals[all_vocals > 0]
    if len(spoken) == 0:
        return "Silent", success_rate
        
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
    for t in range(min(NUM_STEPS, 80)): # Expanded to see the full rollout 
        # FIXED: Removed the [0] batch index because we are only evaluating a single universe here
        p0, p1 = outputs['agent_pos'][t]
        v0, v1 = outputs['vocals'][t]
        h0, h1 = outputs['hunger'][t]
        
        c0, c1 = (chr(AGENT_BASE + v0) if v0 > 0 else '@'), (chr(AGENT_BASE + v1) if v1 > 0 else '@')
        print(f"{t+1:<4} | {p0:02d} / {c0} / {h0:.2f}{'':<6} | {p1:02d} / {c1} / {h1:.2f}{'':<6} | {render_tape_ascii(outputs['tape'][t])}")

# ====================== MAIN ======================
if __name__ == '__main__':
    key = random.PRNGKey(42)
    
    dummy_carry = (jnp.zeros((NUM_AGENTS, HIDDEN_DIM)), jnp.zeros((NUM_AGENTS, HIDDEN_DIM)))
    optimizer = optax.adam(LEARNING_RATE)
    
    def init_model(k):
        return model.init(k, jnp.zeros((NUM_AGENTS, 9), dtype=jnp.int32), 
                          jnp.zeros((NUM_AGENTS, 3)), dummy_carry, 
                          jnp.zeros((NUM_AGENTS, HIDDEN_DIM)), jnp.zeros((NUM_AGENTS,), dtype=jnp.int32), 
                          jnp.zeros((NUM_AGENTS,), dtype=jnp.int32), method=AgentBrain.init_everything)

    key, init_key = random.split(key)
    init_keys = random.split(init_key, NUM_ENVS)
    params_batch = jax.vmap(init_model)(init_keys)
    opt_state_batch = optimizer.init(params_batch)
    
    # NEW: Track "Lifetime Fitness" across many maps, not just the current one
    ema_fitness = jnp.ones(NUM_ENVS) * 1.0 
    
    print(f"🚀 Training Phase 5: UED & Generational Turnover (Envs: {NUM_ENVS})...")
    for epoch in range(NUM_EPOCHS):
        key, env_key, step_key = random.split(key, 3)
        
        env_keys = random.split(env_key, NUM_ENVS)
        state_batch = vectorized_init_env(env_keys)
        
        step_keys = random.split(step_key, NUM_ENVS)
        
        params_batch, opt_state_batch, loss, avg_ext, avg_epi, beta, outputs = update_batch(
            params_batch, opt_state_batch, state_batch, step_keys, dummy_carry, epoch
        )
        
        # Update Lifetime Fitness (90% history, 10% current map)
        current_ext_loss = jnp.mean(outputs['shared_extrinsic'], axis=(0, 1, 2))
        ema_fitness = 0.9 * ema_fitness + 0.1 * current_ext_loss
        
        if epoch > 0 and epoch % 500 == 0:
            key, mut_key = random.split(key)
            
            # Evolve based on Lifetime Fitness
            params_batch, opt_state_batch = evolve_population(params_batch, opt_state_batch, ema_fitness, mut_key)
            
            # Reset the fitness scores for the newborns
            sort_idx = jnp.argsort(ema_fitness)
            top_half_fitness = ema_fitness[sort_idx[:NUM_ENVS // 2]]
            ema_fitness = jnp.concatenate([top_half_fitness, top_half_fitness], axis=0)
            
            print(f"   >>> GENERATIONAL TURNOVER: Culled bottom 50% based on Lifetime Fitness.")
            
        if epoch % 500 == 0:
            top_vocab, win_rate = analyze_language(outputs)
            print(f"Epoch {epoch:4d} | Loss: {float(loss):.4f} | Lifetime Ext: {float(jnp.mean(ema_fitness)):.4f} | Epi: {float(avg_epi):.4f}")
            print(f"   -> Win Rate: {win_rate:.1f}% | Top Vocab: {top_vocab}")
    
    print("\n🎉 Training complete! Greedy test rollout for Universe 0:")
    key, test_key = random.split(key)
    
    # Generate one final random map for the test rollout
    test_state_batch = vectorized_init_env(random.split(test_key, NUM_ENVS))
    test_state = jax.tree_util.tree_map(lambda x: x[0], test_state_batch)
    
    # Pluck the absolute best agent from the population based on Lifetime Fitness
    best_idx = jnp.argmin(ema_fitness)
    best_params = jax.tree_util.tree_map(lambda x: x[best_idx], params_batch)
    
    test_outputs = rollout_single(best_params, test_state, test_key, dummy_carry, beta=0.0, greedy=True)
    print_debug(test_outputs)
