"""Quarl-inspired ZX state-buffer search with optional on-target PPO.

This is not Quarl/HAE: the original ZX action space and actor/critic remain.
Every transition is scored by the same fixed-wire completion used at output.
Fine-tuning, completion, cache and archive maintenance all consume search time.
"""
import contextlib
import copy
import hashlib
import importlib.util
import os
import pickle
import random
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(RESEARCH))
from reconstructed_ppo import batch, data
from cflow_transfer_probe import complete, direct_cflow, nam, cost, clean
from qiskit import qasm2
from buffer import CircuitBuffer, Entry, score, primary, reward
from optimized_backend import rewrite_only_class, ExactCompletion
from inference import take_training_action


METHODS = {
    'quarl_fixed_independent_masks': ('cooperative', True, True),
    'quarl_fixed_shared_masks': ('cooperative', True, True),
    'quarl_fixed_independent_plain': ('cooperative', True, True),
    'quarl_fixed_shared_plain': ('cooperative', True, True),
    'quarl_deferred_train': ('cooperative', True, True),
    'quarl_deferred_random': ('cooperative', False, False),
    'quarl_deferred_frozen': ('cooperative', True, False),
    'restart_uniform': ('restart', False, False),
    'restart_policy': ('restart', True, False),
    'best_uniform': ('best', False, False),
    'best_policy': ('best', True, False),
    'diverse_uniform': ('diverse', False, False),
    'diverse_policy': ('diverse', True, False),
    'diverse_finetune': ('diverse', True, True),
}


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def graph_key(graph):
    # Conservative exact-state deduplication, not graph-isomorphism reduction.
    return hashlib.sha256(pickle.dumps(graph, protocol=5)).hexdigest()


def source_classes(source_dir, optimized=True):
    modules = []
    for name, filename in [('buffer_original_env', 'zx_env.py'),
                           ('buffer_original_agent', 'rl_agent.py')]:
        spec = importlib.util.spec_from_file_location(name, Path(source_dir) / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    Env = rewrite_only_class(modules[0].ZXEnv) if optimized else modules[0].ZXEnv
    return Env, modules[1].AgentGNN


def snapshot(env, obs):
    return copy.deepcopy((env.__dict__, obs))


def restore(env, entry):
    state, obs = copy.deepcopy(entry.payload)
    env.__dict__.clear()
    env.__dict__.update(state)
    # A new trajectory gets a fresh horizon, while retaining graph, match
    # dictionaries and graph observations from the SAME archived state.
    env.episode_len = 0
    env.episode_reward = 0.
    env.episode_stats = dict.fromkeys(env.episode_stats, 0)
    return obs


def distribution(net, obs):
    p = obs[0]
    ids = p.y[p.y != -1]
    if net is None:
        probs = np.full(len(ids), 1. / len(ids))
    else:
        with torch.no_grad():
            probs = torch.softmax(net.actor(p)[p.y != -1].flatten().double(), 0).numpy()
    assert len(ids) and np.isfinite(probs).all()
    return ids, probs


def take_action(net, obs, rng):
    ids, probs = distribution(net, obs)
    slot = min(int(np.searchsorted(np.cumsum(probs), rng.random(), side='right')), len(ids) - 1)
    return slot, int(ids[slot])


def gae(rewards, dones, values, bootstrap, gamma=.99, lam=.95):
    advantages = torch.zeros(len(rewards))
    carry = 0.
    for i in reversed(range(len(rewards))):
        following = bootstrap if i == len(rewards) - 1 else values[i + 1]
        mask = 1. - float(dones[i])
        delta = rewards[i] + gamma * following * mask - values[i]
        carry = delta + gamma * lam * mask * carry
        advantages[i] = carry
    returns = advantages + torch.tensor(values)
    normalized = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    return normalized, returns


def ppo_update(net, optimizer, rollout, bootstrap, config, deadline):
    observations, actions, old_logs, values, rewards, dones = zip(*rollout)
    advantages, returns = gae(rewards, dones, values, bootstrap)
    acts = torch.tensor(actions)
    old_logs = torch.tensor(old_logs)
    losses = []
    before = torch.cat([p.detach().flatten() for p in net.parameters()]).clone()
    actor_before = torch.cat([p.detach().flatten() for p in net.actor_gnn.parameters()]).clone()
    for _ in range(config['epochs']):
        order = torch.randperm(len(rollout))
        for start in range(0, len(rollout), config['minibatch']):
            if time.perf_counter() >= deadline:
                break
            ix = order[start:start + config['minibatch']]
            _, logs, entropy, v, _, _ = net.get_action_and_value(
                batch([observations[i] for i in ix.tolist()]), action=acts[ix], device='cpu')
            ratio = (logs - old_logs[ix]).exp()
            policy_loss = torch.maximum(-advantages[ix] * ratio,
                                        -advantages[ix] * ratio.clamp(.8, 1.2)).mean()
            loss = policy_loss + .5 * (v - returns[ix]).square().mean() - .01 * entropy.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite PPO loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(net.parameters(), .5)
            if not torch.isfinite(grad):
                raise FloatingPointError('nonfinite PPO gradient')
            optimizer.step()
            losses.append(float(loss.detach()))
        if time.perf_counter() >= deadline:
            break
    after = torch.cat([p.detach().flatten() for p in net.parameters()])
    actor_after = torch.cat([p.detach().flatten() for p in net.actor_gnn.parameters()])
    return dict(samples=len(rollout), optimizer_steps=len(losses),
                mean_loss=float(np.mean(losses)) if losses else None,
                parameter_l2_delta=float(torch.linalg.vector_norm(after - before)),
                actor_parameter_l2_delta=float(torch.linalg.vector_norm(actor_after - actor_before)))


class CompletionCache:
    def __init__(self, limit=256, optimized=True, circuit_cache=True, direct_conversion=True,
                 fast_flow=True, verify_flow=False, record_requests=False):
        self.entries = OrderedDict()
        self.limit = limit
        self.hits = self.misses = 0
        self.graph_requests = [] if record_requests else None
        self.scorer = ExactCompletion(clean, nam, cost, limit, circuit_cache, direct_conversion,
                                      fast_flow, verify_flow, record_requests) if optimized else None

    def evaluate(self, graph):
        key = graph_key(graph)
        if self.graph_requests is not None:
            self.graph_requests.append(key)
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return key, self.entries[key], True
        self.misses += 1
        if self.scorer is not None:
            result = self.scorer.evaluate(graph)
        else:
            circuit, _ = complete(graph)
            text = qasm2.dumps(circuit) + '\n'
            result = dict(qasm=text, sha256=digest(text), cost=cost(circuit))
        self.entries[key] = result
        if len(self.entries) > self.limit:
            self.entries.popitem(last=False)
        return key, result, False


def worker(conn, spec, method, checkpoint_path, source_dir, config, search_seed):
    if method.startswith('quarl_'):
        from cooperative import worker as cooperative_worker
        return cooperative_worker(conn, spec, method, checkpoint_path, source_dir, config, search_seed)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(search_seed)
        random.seed(spec['seed'])
        np.random.seed(spec['seed'])
        optimized = config.get('backend', 'optimized') == 'optimized'
        Env, Agent = source_classes(source_dir, optimized=optimized)
        env = Env(spec['qubits'], spec['depth'])
        env.device = 'cpu'
        env.gate_type = 'gates' if config['objective'] == 'nam' else 'twoqubits'
        null = open(os.devnull, 'w')
        with contextlib.redirect_stdout(null):
            _, info = env.reset()
        obs = data(info['graph_obs'])
        source = qasm2.dumps(nam(env.rand_circuit)) + '\n'
        cache = CompletionCache(config['cache_capacity'], optimized=optimized)
        key, root_result, _ = cache.evaluate(env.graph)
        root = Entry(key, root_result['cost'], snapshot(env, obs), 0)
        objective = config['objective']
        archive = CircuitBuffer(root, config['buffer_capacity'], objective)
        baseline_options = {'input': source, 'zero_action': root_result['qasm']}
        baseline_errors = {}
        try:
            baseline_options['direct_cflow'] = qasm2.dumps(direct_cflow(env.graph)) + '\n'
        except Exception as exc:
            baseline_errors['direct_cflow'] = repr(exc)
        baselines = {}
        for objective in ['cx', 'nam']:
            choose = (lambda text: score(cost(qasm2.loads(text)), 'cx')) if objective == 'cx' else (
                lambda text: (cost(qasm2.loads(text))['gates'], cost(qasm2.loads(text))['cx']))
            text = min(baseline_options.values(), key=choose)
            baselines[objective] = dict(qasm=text, sha256=digest(text), cost=cost(qasm2.loads(text)))
        mode, learned, finetune = METHODS[method]
        objective = config['objective']
        net = optimizer = None
        if learned:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            assert checkpoint['objective'] == objective, 'Pretraining objective mismatch'
            net = Agent(None, 'cpu', c_hidden=checkpoint['width'], c_hidden_v=checkpoint['width'])
            net.load_state_dict(checkpoint['state_dict'])
            net.eval()
            with torch.no_grad():
                net.get_action_and_value(batch([obs]), action=torch.tensor([0]), device='cpu')
            if finetune:
                optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'], eps=1e-5)
        # Environment/reset and model construction must not alter paired search RNG.
        rng = np.random.default_rng(search_seed)
        torch.manual_seed(search_seed)
        random.seed(search_seed)
        np.random.seed(search_seed)
        initial_parameters = None if net is None else torch.cat(
            [p.detach().flatten() for p in net.parameters()]).clone()
        conn.send(dict(kind='ready', source=source, input_sha256=digest(source),
                       baselines=baselines, baseline_errors=baseline_errors))
        deadline = conn.recv()
        rollout = []
        transition = trajectory = total_distance = 0
        active = False
        start_entry = None
        previous_counts = None
        maximum_distance = 0
        while time.perf_counter() < deadline:
            if not active:
                start_entry = archive.choose(mode, rng)
                obs = restore(env, start_entry)
                previous_counts = start_entry.counts
                total_distance = start_entry.distance
                trajectory += 1
                active = True
            conn.send(dict(kind='stage', stage='transition'))
            old_log = value = None
            if finetune:
                slot, action, old_log, value = take_training_action(net, obs, rng)
            else:
                slot, action = take_action(net, obs, rng)
            old_obs = obs
            with contextlib.redirect_stdout(null):
                _, _, done, truncated, info = env.step(action)
            assert not truncated
            obs = data(info['graph_obs'])
            key, candidate, hit = cache.evaluate(env.graph)
            transition += 1
            total_distance += int(action != env.shape * (env.shape + 1) + 1)
            maximum_distance = max(maximum_distance, total_distance)
            step_reward = reward(previous_counts, candidate['cost'], objective)
            # Permit temporary degradation, but cap gross cost inflation.
            over_cost = primary(candidate['cost'], objective) > config['alpha'] * max(1, primary(root.counts, objective))
            terminal = bool(done or over_cost)
            admitted = False
            if mode != 'restart' and score(candidate['cost'], objective) <= score(start_entry.counts, objective):
                admitted = archive.add(Entry(key, candidate['cost'], snapshot(env, obs), total_distance),
                                       start_entry.counts)
            # Report every fully completed transition, including intermediate
            # states; the parent alone decides eligibility before its deadline.
            conn.send(dict(kind='candidate', **candidate, graph_key=key,
                           transition=transition, trajectory=trajectory,
                           start_key=start_entry.key, start_distance=start_entry.distance,
                           distance=total_distance, action=action, reward=step_reward, objective=objective,
                           previous_cost=previous_counts, terminal=terminal,
                           cache_hit=hit, admitted=admitted, buffer=archive.summary()))
            if finetune:
                rollout.append((old_obs, slot, old_log, value, step_reward, terminal))
            previous_counts = candidate['cost']
            active = not terminal
            if finetune and len(rollout) >= config['rollout']:
                conn.send(dict(kind='stage', stage='ppo_update'))
                bootstrap = 0.
                if active:
                    with torch.no_grad():
                        bootstrap = float(net.get_value(batch([obs])[1])[0])
                result = ppo_update(net, optimizer, rollout, bootstrap, config, deadline)
                rollout = []
                if result['optimizer_steps'] and time.perf_counter() < deadline:
                    checkpoint_file = Path(config['adapted_checkpoint'])
                    temporary = checkpoint_file.with_suffix('.tmp')
                    torch.save(dict(state_dict=net.state_dict(), width=checkpoint['width'],
                                    objective=objective,
                                    adaptation_transitions=transition,
                                    optimizer_state_dict=optimizer.state_dict(),
                                    scope='On-target PPO adaptation; research checkpoint'), temporary)
                    os.replace(temporary, checkpoint_file)
                if time.perf_counter() < deadline:
                    conn.send(dict(kind='update', **result, transition=transition))
        delta = None if net is None else float(torch.linalg.vector_norm(torch.cat(
            [p.detach().flatten() for p in net.parameters()]) - initial_parameters))
        conn.send(dict(kind='finished', transitions=transition, trajectories=trajectory,
                       max_distance=maximum_distance, buffer=archive.summary(),
                       cache_hits=cache.hits, cache_misses=cache.misses,
                       model_parameter_l2_delta=delta))
    except (BrokenPipeError, EOFError):
        pass
    except BaseException:
        try:
            conn.send(dict(kind='error', traceback=traceback.format_exc()))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        conn.close()
