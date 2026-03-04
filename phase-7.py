import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
import optax
from typing import NamedTuple
import numpy as np

# ====================== CONSTANTS ======================
NUM_ENVS = 1024
NUM_AGENTS = 6              # Scaled up for tribal dynamics
TAPE_SIZE = 256             # Scaled up for long-horizon planning
HIDDEN_DIM = 128            # Increased capacity
NUM_STEPS = 300             # Longer horizons
LEARNING_RATE = 0.0005      # Lowered slightly for Actor-Critic stability
NUM_EPOCHS = 5000
ALPHA = 0.01                # Entropy bonus
GAMMA = 0.99                # Discount factor
LAMBDA = 0.95               # GAE Lambda

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
    done: jnp.ndarray        # NEW: Tracks if the episode is finished for this universe

def init_env_single(key: jnp.ndarray) -> EnvState:
    k1, k2, k3, k4, k5 = random.split(key, 5)
    
    # Extreme UED: 0-3 decoys, massive wire variance
    num_decoys = random.randint(k1, (), 0, 4)
    real_button_pos = random.randint(k2, (), 10, TAPE_SIZE - 100)
    wire_len = random.randint(k3, (), 20, 80)
    door_pos = real_button_pos + wire_len + 1
    core_pos = door_pos + 1
    
    tape = jnp.full(TAPE_SIZE, EMPTY, dtype=jnp.int32)
    tape = tape.at[0].set(WALL)
    tape = tape.at[-1].set(WALL)
    tape = tape.at[real_button_pos].set(BUTTON)
    
    # Place decoys (simplified logic to ensure they fit)
    decoy_pos = random.randint(k5, (3,), 10, TAPE_SIZE - 20)
    for i in range(3):
        pos = decoy_pos[i]
        valid = (pos < real_button_pos - 2) | (pos > door_pos + 2)
        should_place = (i < num_decoys) & valid
        tape = jnp.where(should_place, tape.at[pos].set(BUTTON), tape)
    
    idx = jnp.arange(TAPE_SIZE)
    is_wire = (idx > real_button_pos) & (idx < door_pos)
    tape = jnp.where(is_wire, WIRE, tape)
    
    tape = tape.at[door_pos].set(DOOR_CLOSED)
    tape = tape.at[core_pos].set(CORE)
    
    # Spawn agents near the left side
    agent_pos = jnp.arange(1, NUM_AGENTS + 1, dtype=jnp.int32) 
    hunger = jnp.ones((NUM_AGENTS,), dtype=jnp.float32)
    vocals = jnp.zeros((NUM_AGENTS,), dtype=jnp.int32)
    last_action = jnp.ones((NUM_AGENTS,), dtype=jnp.int32)
    done = jnp.array(False)
    
    return EnvState(agent_pos, hunger, vocals, last_action, tape, done)

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
    # JAX `.at[].set()` naturally handles overlapping indices (last agent in array wins visual)
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

# ====================== ACTOR-CRITIC BRAIN ======================
class AgentBrain(nn.Module):
    hidden_dim: int = HIDDEN_DIM
    
    def setup(self):
        self.embed_vis = nn.Embed(num_embeddings=256, features=16)
        self.lstm = nn.LSTMCell(features=self.hidden_dim) 
        self.policy_act = nn.Dense(3)
        self.policy_voc = nn.Dense(27)
        self.value_head = nn.Dense(1) # NEW: Critic Head for GAE
        
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
        values = self.value_head(lstm_out).squeeze(-1) # Shape: (NUM_AGENTS,)
        
        return p_logits, v_logits, values, new_carry, lstm_out

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

# ====================== GAE & ROLLOUT ======================
def compute_gae(rewards, values, next_value, dones):
    # Vectorized Generalized Advantage Estimation
    # shapes: rewards(T, A), values(T, A), dones(T)
    v_all = jnp.concatenate([values, next_value[None, :]], axis=0)
    dones_expanded = dones[:, None]
    
    deltas = rewards + GAMMA * v_all[1:] * (1.0 - dones_expanded) - v_all[:-1]
    
    def scan_fn(carry, step_data):
        delta, done = step_data
        gae = delta + GAMMA * LAMBDA * (1.0 - done) * carry
        return gae, gae

    _, advantages = jax.lax.scan(scan_fn, jnp.zeros_like(rewards[0]), (deltas[::-1], dones_expanded[::-1]))
    advantages = advantages[::-1]
    returns = advantages + values
    return advantages, returns

def rollout_single(params, state, key, dummy_carry, greedy=False):
    def step(carry_tuple, _):
        state, key, lstm_c = carry_tuple
        tape_with_agents = render_agents_single(state.tape, state.agent_pos, state.vocals)
        vision = get_vision_single(state.agent_pos, tape_with_agents)
        proprio = jnp.stack([state.hunger, state.last_action.astype(jnp.float32), state.vocals.astype(jnp.float32)], axis=-1)
        
        p_logits, v_logits, values, new_lstm_c, h = model.apply(params, vision, proprio, lstm_c, method=AgentBrain.__call__)
        
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
        
        # Mask actions if the episode is already done (Early Termination logic)
        actions = jnp.where(state.done, 1, actions) # 1 is 'Stay'
        vocals = jnp.where(state.done, 0, vocals)   # 0 is 'Silent'
        
        moves = actions - 1 
        intended_pos = jnp.clip(state.agent_pos + moves, 0, TAPE_SIZE - 1)
        blocked = (state.tape[intended_pos] == DOOR_CLOSED) | (state.tape[intended_pos] == WALL)
        new_pos = jnp.where(blocked, state.agent_pos, intended_pos)
        
        new_tape = apply_ca_physics_single(state.tape, new_pos, vocals)
        
        at_core = (new_tape[new_pos] == CORE)
        anyone_ate_core = jnp.any(at_core)
        new_done = state.done | anyone_ate_core
        new_tape = jnp.where(anyone_ate_core, jnp.where(new_tape == CORE, EMPTY, new_tape), new_tape)
        
        # Metabolic tax: only applied if not done
        metabolic_tax = jnp.where((vocals > 0) & ~state.done, 0.01, 0.0)
        base_decay = jnp.where(~state.done, 0.01, 0.0)
        new_hunger = jnp.clip(state.hunger + base_decay + metabolic_tax, 0.0, 1.0)
        new_hunger = jnp.where(anyone_ate_core, 0.0, new_hunger)
        
        # Reward Shaping
        step_reward = -new_hunger
        # Huge terminal bonus for speed + efficiency
        bonus = jnp.where(anyone_ate_core & ~state.done, 10.0 + (1.0 - jnp.mean(new_hunger)) * 5.0, 0.0)
        step_reward = step_reward + bonus
        
        new_state = EnvState(new_pos, new_hunger, vocals, actions, new_tape, new_done)
        
        # Epistemic calculation
        h_sg = jax.lax.stop_gradient(h)
        pred_vision_logits = model.apply(params, h_sg, actions, vocals, method=AgentBrain.predict_next)
        new_tape_with_agents = render_agents_single(new_state.tape, new_state.agent_pos, new_state.vocals)
        new_vision = get_vision_single(new_state.agent_pos, new_tape_with_agents)
        ce_loss = optax.softmax_cross_entropy_with_integer_labels(pred_vision_logits, new_vision)
        epistemic = jnp.clip(jnp.mean(ce_loss, axis=-1), 0.0, 2.0)
        
        output = {
            'p_logits': p_logits, 'v_logits': v_logits, 'actions': actions, 'vocals': vocals, 
            'values': values, 'rewards': step_reward, 'dones': state.done, 'entropy': entropy,
            'epistemic': epistemic, 'hunger': new_state.hunger, 'tape': new_tape_with_agents,
            'agent_pos': new_state.agent_pos
        }
        return (new_state, key, new_lstm_c), output
    
    (final_state, key, final_lstm_c), outputs = jax.lax.scan(step, (state, key, dummy_carry), jnp.arange(NUM_STEPS))
    
    # Get the bootstrap value for the very last step for GAE
    tape_with_agents = render_agents_single(final_state.tape, final_state.agent_pos, final_state.vocals)
    vision = get_vision_single(final_state.agent_pos, tape_with_agents)
    proprio = jnp.stack([final_state.hunger, final_state.last_action.astype(jnp.float32), final_state.vocals.astype(jnp.float32)], axis=-1)
    _, _, next_value, _, _ = model.apply(params, vision, proprio, final_lstm_c, method=AgentBrain.__call__)
    
    outputs['next_value'] = next_value
    return outputs

def loss_fn_single(p, state, key, dummy_carry, beta):
    outputs = rollout_single(p, state, key, dummy_carry, greedy=False)
    
    # Calculate GAE
    advantages, returns = compute_gae(outputs['rewards'], outputs['values'], outputs['next_value'], outputs['dones'])
    
    # Normalize advantages for stable policy updates
    adv_mean = jnp.mean(advantages)
    adv_std = jnp.std(advantages) + 1e-8
    norm_advantages = (advantages - adv_mean) / adv_std
    
    p_log_probs, v_log_probs = jax.nn.log_softmax(outputs['p_logits']), jax.nn.log_softmax(outputs['v_logits'])
    chosen_p = jnp.take_along_axis(p_log_probs, outputs['actions'][..., None], axis=-1).squeeze(-1)
    chosen_v = jnp.take_along_axis(v_log_probs, outputs['vocals'][..., None], axis=-1).squeeze(-1)
    
    # Actor-Critic Loss Components
    policy_loss = -jnp.mean((chosen_p + chosen_v) * jax.lax.stop_gradient(norm_advantages))
    value_loss = 0.5 * jnp.mean(jnp.square(returns - outputs['values']))
    transition_loss = jnp.mean(outputs['epistemic'])
    entropy_bonus = -ALPHA * jnp.mean(outputs['entropy'])
    
    total_loss = policy_loss + value_loss + transition_loss + entropy_bonus
    
    # Fitness metric: Mean final hunger of the universe
    final_hunger_score = jnp.mean(outputs['hunger'][-1])
    return total_loss, (final_hunger_score, transition_loss)

# ====================== P-MAP & EVOLUTION ======================
def update_device_raw(params_shard, opt_state_shard, state_shard, key_shard, dummy_carry, beta):
    vg_fn = jax.vmap(jax.value_and_grad(loss_fn_single, has_aux=True), in_axes=(0, 0, 0, None, None))
    (loss_batch, (ext_batch, epi_batch)), grads_batch = vg_fn(params_shard, state_shard, key_shard, dummy_carry, beta)
    
    grads_synced = jax.lax.pmean(grads_batch, axis_name='i')
    updates, new_opt_state = optimizer.update(grads_synced, opt_state_shard, params_shard)
    new_params = optax.apply_updates(params_shard, updates)
    
    return new_params, new_opt_state, jnp.mean(loss_batch), ext_batch, jnp.mean(epi_batch)

update_device = jax.pmap(update_device_raw, axis_name='i', in_axes=(0, 0, 0, 0, None, None))

def evolve_global_population(params, opt_state, ema_fitness, key):
    flat_fitness = ema_fitness.reshape(NUM_ENVS)
    sort_idx = jnp.argsort(flat_fitness)
    top_idx = sort_idx[:NUM_ENVS // 2]
    
    def apply_evolution(x):
        if x.ndim == 1 and x.shape[0] == num_devices: return x
        flat_x = x.reshape((NUM_ENVS,) + x.shape[2:])
        top_half = flat_x[top_idx]
        cloned = jnp.concatenate([top_half, top_half], axis=0)
        return cloned.reshape((num_devices, envs_per_device) + x.shape[2:])

    new_params = jax.tree_util.tree_map(apply_evolution, params)
    new_opt_state = jax.tree_util.tree_map(apply_evolution, opt_state)
    
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

# ====================== DIAGNOSTICS ======================
def render_tape_ascii(tape_array):
    char_map = {EMPTY: '.', BUTTON: '_', BUTTON_ON: '=', WIRE: 'w', WIRE_ON: '*', 
                DOOR_CLOSED: 'd', DOOR_OPEN: '-', CORE: 'z', WALL: '#'}
    return '[' + "".join([char_map.get(int(b), chr(b) if 64 <= b <= 90 else '?') for b in tape_array]) + ']'

def print_debug(outputs):
    print(f"{'Tick':<4} | {'A0 Pos/Voc':<12} | {'A1 Pos/Voc':<12} | {'Tape'}")
    print('-' * 110)
    
    for t in range(min(NUM_STEPS, 150)): 
        # Skip printing if they've been done for a while to save space
        if t > 0 and outputs['dones'][t] and outputs['dones'][t-1]:
            continue
            
        p0, p1 = outputs['agent_pos'][t][0], outputs['agent_pos'][t][1]
        v0, v1 = outputs['vocals'][t][0], outputs['vocals'][t][1]
        c0, c1 = (chr(AGENT_BASE + v0) if v0 > 0 else '@'), (chr(AGENT_BASE + v1) if v1 > 0 else '@')
        
        # Sliced tape rendering so it fits on screen
        tape_str = render_tape_ascii(outputs['tape'][t])
        
        # Center the view roughly around the action
        view_start = max(0, min(p0, p1) - 10)
        view_end = min(TAPE_SIZE, max(p0, p1) + 40)
        visible_tape = tape_str[view_start:view_end]
        
        print(f"{t+1:<4} | {p0:03d} / {c0:<6} | {p1:03d} / {c1:<6} | ...{visible_tape}...")

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
    sharded_opt_state = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (num_devices,)) if x.ndim == 0 else x, 
        sharded_opt_state
    )
    
    ema_fitness = jnp.ones((num_devices, envs_per_device)) * 1.0 
    
    print(f"🚀 PHASE 7 LAUNCH: Actor-Critic + GAE + 256-tile 1D + {NUM_AGENTS} Agents")
    for epoch in range(NUM_EPOCHS):
        key, env_key, step_key = random.split(key, 3)
        
        flat_envs = vectorized_init_env(random.split(env_key, NUM_ENVS))
        sharded_envs = jax.tree_util.tree_map(lambda x: x.reshape((num_devices, envs_per_device) + x.shape[1:]), flat_envs)
        sharded_step_keys = random.split(step_key, NUM_ENVS).reshape((num_devices, envs_per_device, 2))
        
        # Beta schedule for epistemic curiosity
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
            
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Loss: {float(jnp.mean(loss)):.4f} | Lifetime Ext (Hunger): {float(jnp.mean(ema_fitness)):.4f}")
    
    print("\n🎉 Training complete! Running diagnostics rollout...")
    key, test_key = random.split(key)
    
    flat_fitness = ema_fitness.reshape(NUM_ENVS)
    best_idx = jnp.argmin(flat_fitness)
    device_idx, env_idx = best_idx // envs_per_device, best_idx % envs_per_device
    best_params = jax.tree_util.tree_map(lambda x: x[device_idx, env_idx], sharded_params)
    
    test_state = init_env_single(test_key)
    test_outputs = rollout_single(best_params, test_state, test_key, dummy_carry, greedy=True)
    
    print_debug(test_outputs)
