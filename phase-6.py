
# *** Be sure to enable the TPU v5e-8 as the Accelorator on the right side menu in Kaggle, not the P100 GPU, for this phase. ***

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
TAPE_SIZE = 40          # Expanded for complex puzzles
HIDDEN_DIM = 64
NUM_STEPS = 100         # Expanded for longer travel times
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

# ====================== TPU SETUP & SHARDING ======================
num_devices = jax.device_count()
envs_per_device = NUM_ENVS // num_devices
print(f"🔥 Detected {num_devices} TPU/GPU devices. Sharding {envs_per_device} universes per device.")

# ====================== ADVANCED UED ENVIRONMENT ======================
class EnvState(NamedTuple):
    agent_pos: jnp.ndarray   
    hunger: jnp.ndarray      
    vocals: jnp.ndarray      
    last_action: jnp.ndarray 
    tape: jnp.ndarray        

def init_env_single(key: jnp.ndarray) -> EnvState:
    k1, k2, k3, k4 = random.split(key, 4)
    
    # Advanced UED: Decoy buttons and random wires
    b1_pos = random.randint(k1, (), 2, 6)
    b2_pos = random.randint(k2, (), 8, 12)
    
    # Randomly choose which button is the REAL one connected to the wire
    is_b1_real = random.uniform(k3) > 0.5
    real_button_pos = jnp.where(is_b1_real, b1_pos, b2_pos)
    
    wire_len = random.randint(k4, (), 4, 12)
    door_pos = real_button_pos + wire_len + 1
    core_pos = door_pos + 1
    
    tape = jnp.full(TAPE_SIZE, EMPTY, dtype=jnp.int32)
    tape = tape.at[0].set(WALL)
    tape = tape.at[-1].set(WALL)
    tape = tape.at[b1_pos].set(BUTTON)
    tape = tape.at[b2_pos].set(BUTTON)
    
    idx = jnp.arange(TAPE_SIZE)
    is_wire = (idx > real_button_pos) & (idx < door_pos)
    tape = jnp.where(is_wire, WIRE, tape)
    
    tape = tape.at[door_pos].set(DOOR_CLOSED)
    tape = tape.at[core_pos].set(CORE)
    
    agent_pos = jnp.array([1, 2], dtype=jnp.int32) 
    hunger = jnp.ones((NUM_AGENTS,), dtype=jnp.float32)
    vocals = jnp.zeros((NUM_AGENTS,), dtype=jnp.int32)
    last_action = jnp.ones((NUM_AGENTS,), dtype=jnp.int32)
    
    return EnvState(agent_pos, hunger, vocals, last_action, tape)

vectorized_init_env = jax.vmap(init_env_single, in_axes=(0,))

def apply_ca_physics_single(tape, agent_pos, vocals):
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
    
    blocked = (tape[intended_pos] == DOOR_CLOSED) | (tape[intended_pos] == WALL)
    new_pos = jnp.where(blocked, pos, intended_pos)
    new_tape = apply_ca_physics_single(tape, new_pos, vocals)
    
    at_core = (new_tape[new_pos] == CORE)
    anyone_ate_core = jnp.any(at_core)
    new_tape = jnp.where(anyone_ate_core, jnp.where(new_tape == CORE, EMPTY, new_tape), new_tape)
    
    metabolic_tax = jnp.where(vocals > 0, 0.01, 0.0) 
    new_hunger = jnp.clip(hunger + 0.01 + metabolic_tax, 0.0, 1.0)
    new_hunger = jnp.where(anyone_ate_core, 0.0, new_hunger)
    return new_pos, new_hunger, new_tape

# ====================== HYPERSCALE BRAIN ======================
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

model = AgentBrain()

# ====================== DISTRIBUTED ROLLOUT & UPDATE ======================
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
    # Returns the scalar loss, plus the actual extrinsic score (shape 128) of this universe
    return total_loss, (jnp.mean(outputs['shared_extrinsic']), jnp.mean(outputs['shared_epistemic']))

def update_device_raw(params_shard, opt_state_shard, state_shard, key_shard, dummy_carry, beta):
    vg_fn = jax.vmap(jax.value_and_grad(loss_fn_single, has_aux=True), in_axes=(0, 0, 0, None, None))
    (loss_batch, (ext_batch, epi_batch)), grads_batch = vg_fn(params_shard, state_shard, key_shard, dummy_carry, beta)
    
    grads_synced = jax.lax.pmean(grads_batch, axis_name='i')
    updates, new_opt_state = optimizer.update(grads_synced, opt_state_shard, params_shard)
    new_params = optax.apply_updates(params_shard, updates)
    
    # Return ext_batch directly so we can track exact lifetime fitness
    return new_params, new_opt_state, jnp.mean(loss_batch), ext_batch, jnp.mean(epi_batch)

update_device = jax.pmap(update_device_raw, axis_name='i', in_axes=(0, 0, 0, 0, None, None))

def evolve_global_population(params, opt_state, ema_fitness, key):
    flat_fitness = ema_fitness.reshape(NUM_ENVS)
    sort_idx = jnp.argsort(flat_fitness)
    top_idx = sort_idx[:NUM_ENVS // 2]
    
    def apply_evolution(x):
        # Ignore scalar step counters - return them exactly as they are
        if x.ndim == 1 and x.shape[0] == num_devices: 
            return x
            
        flat_x = x.reshape((NUM_ENVS,) + x.shape[2:])
        top_half = flat_x[top_idx]
        cloned = jnp.concatenate([top_half, top_half], axis=0)
        return cloned.reshape((num_devices, envs_per_device) + x.shape[2:])

    new_params = jax.tree_util.tree_map(apply_evolution, params)
    new_opt_state = jax.tree_util.tree_map(apply_evolution, opt_state)
    
    # Only apply noise to the actual neural network parameters, NOT the opt_state!
    def apply_noise(p, k):
        noise = jax.random.normal(k, p.shape) * 0.05
        mask = jnp.concatenate([jnp.zeros(NUM_ENVS // 2), jnp.ones(NUM_ENVS // 2)])
        mask = mask.reshape((num_devices, envs_per_device) + (1,) * (p.ndim - 2))
        return p + (noise * mask)

    leaves, treedef = jax.tree_util.tree_flatten(new_params)
    keys = jax.random.split(key, len(leaves))
    noisy_leaves = [apply_noise(leaf, k) for leaf, k in zip(leaves, keys)]
    final_params = jax.tree_util.tree_unflatten(treedef, noisy_leaves)
    
    return final_params, new_opt_state

# ====================== THE "GOD" LLM TRANSLATOR ======================
def generate_llm_translation_prompt(outputs):
    prompt = "I am training a multi-agent AI system using Deep Active Inference. They are navigating a 1D tape.\n"
    prompt += "The agents cannot see the whole tape, only a 9-tile local window. They must coordinate to find the correct button '_', press it to send electricity '*' down a wire 'w', open a door 'd', and eat a core 'z'. They also must avoid decoy buttons.\n\n"
    prompt += "Here is the timeline of a successful run. Analyze the agents' communications (Agent 0 and Agent 1 Vocals). What letters did they use, and what semantic meaning or command does that letter seem to represent based on the physical timing?\n\n"
    
    for t in range(min(NUM_STEPS, 80)): 
        p0, p1 = outputs['agent_pos'][t]
        v0, v1 = outputs['vocals'][t]
        c0, c1 = (chr(AGENT_BASE + v0) if v0 > 0 else '@'), (chr(AGENT_BASE + v1) if v1 > 0 else '@')
        tape_str = render_tape_ascii(outputs['tape'][t])
        
        if v0 > 0 or v1 > 0 or '-' in tape_str:
            prompt += f"Tick {t+1}: Tape: {tape_str} | Agent 0 (Pos {p0:02d}) said: {c0} | Agent 1 (Pos {p1:02d}) said: {c1}\n"
    
    return prompt

def render_tape_ascii(tape_array):
    char_map = {EMPTY: '.', BUTTON: '_', BUTTON_ON: '=', WIRE: 'w', WIRE_ON: '*', 
                DOOR_CLOSED: 'd', DOOR_OPEN: '-', CORE: 'z', WALL: '#'}
    return '[' + "".join([char_map.get(int(b), chr(b) if 64 <= b <= 90 else '?') for b in tape_array]) + ']'

def print_debug(outputs):
    print(f"{'Tick':<4} | {'A0 (Pos/Voc/Hunger)':<20} | {'A1 (Pos/Voc/Hunger)':<20} | {'Tape'}")
    print('-' * 95)
    for t in range(min(NUM_STEPS, 80)): 
        p0, p1 = outputs['agent_pos'][t]
        v0, v1 = outputs['vocals'][t]
        h0, h1 = outputs['hunger'][t]
        c0, c1 = (chr(AGENT_BASE + v0) if v0 > 0 else '@'), (chr(AGENT_BASE + v1) if v1 > 0 else '@')
        print(f"{t+1:<4} | {p0:02d} / {c0} / {h0:.2f}{'':<6} | {p1:02d} / {c1} / {h1:.2f}{'':<6} | {render_tape_ascii(outputs['tape'][t])}")

# ====================== MAIN ======================
if __name__ == '__main__':
    key = random.PRNGKey(42)
    optimizer = optax.adam(LEARNING_RATE)
    dummy_carry = (jnp.zeros((NUM_AGENTS, HIDDEN_DIM)), jnp.zeros((NUM_AGENTS, HIDDEN_DIM)))
    
    def init_model(k):
        return model.init(k, jnp.zeros((NUM_AGENTS, 9), dtype=jnp.int32), 
                          jnp.zeros((NUM_AGENTS, 3)), dummy_carry, 
                          jnp.zeros((NUM_AGENTS, HIDDEN_DIM)), jnp.zeros((NUM_AGENTS,), dtype=jnp.int32), 
                          jnp.zeros((NUM_AGENTS,), dtype=jnp.int32), method=AgentBrain.init_everything)

    key, init_key = random.split(key)
    flat_params = jax.vmap(init_model)(random.split(init_key, NUM_ENVS))
    sharded_params = jax.tree_util.tree_map(lambda x: x.reshape((num_devices, envs_per_device) + x.shape[1:]), flat_params)
    
    sharded_opt_state = optimizer.init(sharded_params)
    
    # Broadcast Optax's internal 0-d step counter to shape (8,)
    sharded_opt_state = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (num_devices,)) if x.ndim == 0 else x, 
        sharded_opt_state
    )
    
    ema_fitness = jnp.ones((num_devices, envs_per_device)) * 1.0 
    
    print(f"🚀 Training Phase 6: TPU Hyperscale & Advanced UED (Envs: {NUM_ENVS} across {num_devices} TPUs)...")
    for epoch in range(NUM_EPOCHS):
        key, env_key, step_key = random.split(key, 3)
        
        flat_envs = vectorized_init_env(random.split(env_key, NUM_ENVS))
        sharded_envs = jax.tree_util.tree_map(lambda x: x.reshape((num_devices, envs_per_device) + x.shape[1:]), flat_envs)
        
        sharded_step_keys = random.split(step_key, NUM_ENVS).reshape((num_devices, envs_per_device, 2))
        beta = jnp.maximum(0.001, 0.1 - (epoch / 20000.0))
        
        sharded_params, sharded_opt_state, loss, ext_batch_sharded, avg_epi = update_device(
            sharded_params, sharded_opt_state, sharded_envs, sharded_step_keys, dummy_carry, beta
        )
        
        ema_fitness = 0.9 * ema_fitness + 0.1 * ext_batch_sharded
        
        if epoch > 0 and epoch % 500 == 0:
            key, mut_key = random.split(key)
            sharded_params, sharded_opt_state = evolve_global_population(sharded_params, sharded_opt_state, ema_fitness, mut_key)
            
            flat_fitness = ema_fitness.reshape(NUM_ENVS)
            top_half = flat_fitness[jnp.argsort(flat_fitness)[:NUM_ENVS // 2]]
            ema_fitness = jnp.concatenate([top_half, top_half], axis=0).reshape((num_devices, envs_per_device))
            
            print(f"   >>> GENERATIONAL TURNOVER: Culled bottom 50%. Global TPUs synced.")
            
        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d} | Loss: {float(jnp.mean(loss)):.4f} | Lifetime Ext: {float(jnp.mean(ema_fitness)):.4f}")
    
    print("\n🎉 Training complete! Running LLM Translator Extractor...")
    key, test_key = random.split(key)
    
    flat_fitness = ema_fitness.reshape(NUM_ENVS)
    best_idx = jnp.argmin(flat_fitness)
    device_idx, env_idx = best_idx // envs_per_device, best_idx % envs_per_device
    best_params = jax.tree_util.tree_map(lambda x: x[device_idx, env_idx], sharded_params)
    
    test_state = init_env_single(test_key)
    test_outputs = rollout_single(best_params, test_state, test_key, dummy_carry, beta=0.0, greedy=True)
    
    print_debug(test_outputs)
    
    llm_prompt = generate_llm_translation_prompt(test_outputs)
    print("\n================ LLM TRANSLATOR PROMPT =================\n")
    print(llm_prompt)
    print("\n========================================================")
