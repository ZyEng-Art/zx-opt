"""Cooperatively scheduled FT/search lanes with exact total-gate feedback.

Two independent environments share discovered incumbents. CPU blocks interleave
rather than run concurrently; all lane work, PPO, refresh and exchange is charged.
This retains the flat RL-ZX actor/global GAE, not Quarl's hierarchical actor/HAE.
"""
import contextlib
import copy
import os
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from qiskit import qasm2

from runtime import (source_classes, data, CompletionCache, snapshot, restore,
                     nam, direct_cflow, cost, digest, distribution, take_action,
                     take_training_action, batch, ppo_update)
from buffer import Entry, CircuitBuffer, score, primary, reward
from search_state import BestFrontier, VisitMasks
from diversity import DiverseFrontier, StratifiedBuffer, Diagnostics

OPTIONS = {
    'quarl_fixed_independent_masks': dict(learned=True, train=True, sharing=False, masks=True, deferred=False),
    'quarl_fixed_shared_masks': dict(learned=True, train=True, sharing=True, masks=True, deferred=False),
    'quarl_fixed_independent_plain': dict(learned=True, train=True, sharing=False, masks=False, deferred=False),
    'quarl_fixed_shared_plain': dict(learned=True, train=True, sharing=True, masks=False, deferred=False),
    'quarl_deferred_train': dict(learned=True, train=True, sharing=True, masks=True, deferred=True),
    'quarl_deferred_random': dict(learned=False, train=False, sharing=True, masks=True, deferred=True),
    'quarl_deferred_frozen': dict(learned=True, train=False, sharing=True, masks=True, deferred=True),
}


def search_action(net, obs, rng, key, masks):
    ids, probabilities = distribution(net, obs)
    legal = [int(a) for a in ids]
    if masks is None:
        slots, details = list(range(len(legal))), dict(legal=len(legal), eligible=len(legal))
    else:
        slots, details = masks.eligible(key, legal)
    if not slots:
        return None
    if net is None:
        selected = slots[int(rng.integers(len(slots)))]
    else:
        selected = slots[int(np.argmax(probabilities[slots]))]
    assert selected in slots
    return selected, legal[selected], details


def worker(conn, spec, method, checkpoint_path, source_dir, config, search_seed):
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(search_seed)
        random.seed(spec['seed'])
        np.random.seed(spec['seed'])
        options = OPTIONS[method]
        objective = config['objective']
        assert objective == 'nam', 'Cooperative pilot targets total NAM gates'
        Env, Agent = source_classes(source_dir, optimized=config.get('backend', 'optimized') == 'optimized')
        env = Env(spec['qubits'], spec['depth'])
        env.device, env.gate_type = 'cpu', 'gates'
        null = open(os.devnull, 'w')
        with contextlib.redirect_stdout(null):
            _, info = env.reset()
        obs = data(info['graph_obs'])
        source = qasm2.dumps(nam(env.rand_circuit)) + '\n'
        cache = CompletionCache(config['cache_capacity'])
        key, root_result, _ = cache.evaluate(env.graph)
        root = Entry(key, root_result['cost'], snapshot(env, obs), 0)
        archive_class = StratifiedBuffer if options['deferred'] else CircuitBuffer
        archive = archive_class(root, config['buffer_capacity'], objective)
        frontier = (DiverseFrontier(root, config['buffer_capacity'], objective,
                                   config.get('frontier_slack_fraction', .1)) if options['deferred']
                    else BestFrontier(root, config['buffer_capacity'], objective))
        masks = VisitMasks(config.get('mask_capacity', 256)) if options['masks'] else None
        diagnostics = Diagnostics()
        second_env = Env(spec['qubits'], spec['depth'])
        restore(second_env, root)
        lanes = {k: dict(env=e, active=False, reason='initial') for k,e in [('ft',env),('pgs',second_env)]}
        baseline_options = {'input': source, 'zero_action': root_result['qasm']}
        errors = {}
        try:
            baseline_options['direct_cflow'] = qasm2.dumps(direct_cflow(env.graph)) + '\n'
        except Exception as exc:
            errors['direct_cflow'] = repr(exc)
        baselines = {}
        for name in ['nam', 'cx']:
            text = min(baseline_options.values(), key=lambda text: score(cost(qasm2.loads(text)), name))
            baselines[name] = dict(qasm=text, sha256=digest(text), cost=cost(qasm2.loads(text)))
        net = search_net = optimizer = None
        if options['learned']:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            assert checkpoint['objective'] == objective
            net = Agent(None, 'cpu', c_hidden=checkpoint['width'], c_hidden_v=checkpoint['width'])
            net.load_state_dict(checkpoint['state_dict'])
            net.eval()
            search_net = copy.deepcopy(net)
            with torch.no_grad():
                net.get_action_and_value(batch([obs]), action=torch.tensor([0]), device='cpu')
            if options['train']:
                optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'], eps=1e-5)
        rng = np.random.default_rng(search_seed)
        torch.manual_seed(search_seed)
        random.seed(search_seed)
        np.random.seed(search_seed)
        initial = None if net is None else torch.cat([p.detach().flatten() for p in net.parameters()]).clone()
        conn.send(dict(kind='ready', source=source, input_sha256=digest(source),
                       baselines=baselines, baseline_errors=errors))
        # Python3.9/macOS perf_counter origins differ across processes. Parent
        # still enforces the exact acceptance deadline; derive the worker's
        # cooperative window from a duration, never its parent's absolute time.
        conn.recv()
        started = time.perf_counter()
        deadline = started + config['budget_seconds']
        ft_version = pgs_version = 0
        transition = trajectory = scheduled = 0
        exchanges = {'ft_to_pgs': 0, 'pgs_to_ft': 0}
        block = config.get('collaboration_block', 8)
        refresh_seconds = config.get('model_refresh_seconds', 5.)
        refresh_tick = 0
        rollout = []
        lane_steps = {'ft': 0, 'pgs': 0}
        while time.perf_counter() < deadline:
            which = 'ft' if (scheduled // block) % 2 == 0 else 'pgs'
            scheduled += 1
            refresh = None
            tick = int((time.perf_counter() - started) // refresh_seconds)
            if which == 'pgs' and tick > refresh_tick:
                refresh_tick = tick
                # Clock-based notification only. Never reset trajectories/masks
                # or refresh early in response to a discovered state.
                changed = search_net is not None and ft_version > pgs_version
                if changed:
                    t = time.perf_counter()
                    search_net.load_state_dict(net.state_dict())
                    pgs_version = ft_version
                    diagnostics.seconds['model_refresh'] += time.perf_counter() - t
                    refresh = 'fixed_clock_model_refresh'
                conn.send(dict(kind='refresh', tick=tick, due_seconds=tick*refresh_seconds,
                               observed_seconds=time.perf_counter()-started, changed=changed,
                               ft_model_version=ft_version, pgs_model_version=pgs_version,
                               interrupted=False, masks_reset=False))
            lane = lanes[which]
            if not lane['active']:
                t = time.perf_counter()
                entry = archive.choose('diverse', rng) if which == 'ft' else frontier.choose(rng, masks)
                if entry is None:
                    diagnostics.blocked += 1
                    which, lane = 'ft', lanes['ft']
                    if lane['active']:
                        entry = None
                    else:
                        entry = archive.choose('diverse', rng)
                if entry is not None:
                    diagnostics.start(which, entry, lane['reason'])
                    lane.update(obs=restore(lane['env'], entry), start=entry, previous=entry.counts,
                                key=entry.key, distance=entry.distance, active=True)
                    trajectory += 1
                    lane['trajectory'] = trajectory
                diagnostics.seconds['restart'] += time.perf_counter() - t
            env, obs = lane['env'], lane['obs']
            old_obs, old_key = obs, lane['key']
            old_log = value = None
            details = None
            conn.send(dict(kind='stage', stage=which + '_transition'))
            t = time.perf_counter()
            if which == 'ft':
                if options['train']:
                    slot, action, old_log, value = take_training_action(net, obs, rng)
                else:
                    slot, action = take_action(net, obs, rng)
            else:
                selection = search_action(search_net, obs, rng, old_key, masks)
                if selection is None:
                    lane['active'] = False
                    lane['reason'] = 'mask_exhausted'
                    diagnostics.blocked += 1
                    diagnostics.seconds['action'] += time.perf_counter() - t
                    continue
                slot, action, details = selection
            diagnostics.seconds['action'] += time.perf_counter() - t
            t = time.perf_counter()
            with contextlib.redirect_stdout(null):
                _, _, done, truncated, info = env.step(action)
            assert not truncated
            obs = data(info['graph_obs'])
            diagnostics.seconds['env_step'] += time.perf_counter() - t
            t = time.perf_counter()
            key, candidate, hit = cache.evaluate(env.graph)
            diagnostics.seconds['completion'] += time.perf_counter() - t
            diagnostics.observe(which, old_key, key, action)
            previous = lane['previous']
            step_reward = reward(previous, candidate['cost'], objective)
            stop = action == env.shape * (env.shape + 1) + 1
            over_cost = primary(candidate['cost'], objective) > config['alpha'] * max(1, primary(root.counts, objective))
            terminal = bool(done or over_cost)
            distance = lane['distance'] + int(not stop)
            if which == 'pgs' and masks is not None:
                masks.record(old_key, action, hard=bool(stop or over_cost or key == old_key))
            entry = None
            eligible_archive = score(candidate['cost'], objective) <= score(lane['start'].counts, objective)
            eligible_frontier = (frontier.eligible(candidate['cost']) if options['deferred'] else
                                 score(candidate['cost'], objective) <= frontier.best)
            t = time.perf_counter()
            if (which == 'ft' and eligible_archive) or eligible_frontier:
                entry = Entry(key, candidate['cost'], snapshot(env, obs), distance)
            admitted = False
            if which == 'ft' and eligible_archive:
                admitted = archive.add(entry, lane['start'].counts)
            exchanged = None
            improved = False
            frontier_added = False
            if (options['sharing'] or which == 'pgs') and eligible_frontier:
                new_key = key not in frontier.entries
                improved = frontier.offer(entry)
                frontier_added = new_key and key in frontier.entries
                diagnostics.offers[which + '_accepted'] += int(frontier_added)
                if improved:
                    if not options['deferred']:
                        if lanes['pgs']['active']:
                            diagnostics.interruptions['new_best_from_' + which] += 1
                        lanes['pgs']['active'] = False
                        lanes['pgs']['reason'] = 'new_best_from_' + which
                        if masks is not None:
                            masks.restart()
                if options['sharing'] and (improved or (options['deferred'] and which == 'ft' and frontier_added)):
                    direction = 'ft_to_pgs' if which == 'ft' else 'pgs_to_ft'
                    exchanges[direction] += 1
                    injected = archive.add(entry, root.counts) if which == 'pgs' else False
                    exchanged = dict(direction=direction, generation=frontier.generation,
                                     archive_injected=injected, score=list(frontier.best),
                                     strict_best=improved, interrupted=not options['deferred'])
            diagnostics.seconds['pool'] += time.perf_counter() - t
            interrupted = which == 'pgs' and improved and not options['deferred']
            lane.update(obs=obs, key=key, previous=candidate['cost'], distance=distance,
                        active=bool(not terminal and not interrupted))
            if terminal:
                lane['reason'] = 'stop' if stop else ('cost_cap' if over_cost else 'horizon')
            transition += 1
            lane_steps[which] += 1
            if which == 'ft' and options['train']:
                rollout.append((old_obs, slot, old_log, value, step_reward, terminal))
            conn.send(dict(kind='candidate', **candidate, graph_key=key, old_graph_key=old_key, lane=which,
                           transition=transition, trajectory=lane['trajectory'],
                           start_key=lane['start'].key, start_distance=lane['start'].distance,
                           distance=distance, action=action, reward=step_reward, objective=objective,
                           previous_cost=previous, terminal=terminal, cache_hit=hit, admitted=admitted,
                           buffer=archive.summary(), frontier_score=list(frontier.best),
                           frontier_generation=frontier.generation, state_exchange=exchanged,
                           exchange_totals=dict(exchanges), mask_selection=details,
                           ft_model_version=ft_version, pgs_model_version=pgs_version,
                           model_refresh=refresh, refresh_tick=refresh_tick, lane_steps=dict(lane_steps),
                           frontier_size=len(frontier.entries), frontier_added=frontier_added,
                           diagnostics=diagnostics.summary()))
            if which == 'ft' and options['train'] and len(rollout) >= config['rollout']:
                t = time.perf_counter()
                conn.send(dict(kind='stage', stage='ft_ppo_update'))
                bootstrap = 0.
                if lane['active']:
                    with torch.no_grad():
                        bootstrap = float(net.get_value(batch([obs])[1])[0])
                result = ppo_update(net, optimizer, rollout, bootstrap, config, deadline)
                rollout = []
                if result['optimizer_steps']:
                    ft_version += 1
                if result['optimizer_steps'] and time.perf_counter() < deadline:
                    checkpoint_file = Path(config['adapted_checkpoint'])
                    temporary = checkpoint_file.with_suffix('.tmp')
                    torch.save(dict(state_dict=net.state_dict(), width=checkpoint['width'], objective=objective,
                                    optimizer_state_dict=optimizer.state_dict(), ft_model_version=ft_version,
                                    adaptation_transitions=lane_steps['ft'],
                                    scope='On-target FT; cooperative search; research weights'), temporary)
                    os.replace(temporary, checkpoint_file)
                diagnostics.seconds['ppo_and_checkpoint'] += time.perf_counter() - t
                if time.perf_counter() < deadline:
                    conn.send(dict(kind='update', **result, transition=transition,
                                   ft_model_version=ft_version, pgs_model_version=pgs_version,
                                   diagnostics=diagnostics.summary()))
        delta = None if net is None else float(torch.linalg.vector_norm(
            torch.cat([p.detach().flatten() for p in net.parameters()]) - initial))
        conn.send(dict(kind='finished', transitions=transition, lane_steps=lane_steps,
                       exchanges=exchanges, buffer=archive.summary(), frontier=list(frontier.best),
                       model_parameter_l2_delta=delta, ft_model_version=ft_version,
                       pgs_model_version=pgs_version, diagnostics=diagnostics.summary()))
    except (BrokenPipeError, EOFError):
        pass
    except BaseException:
        try:
            conn.send(dict(kind='error', traceback=traceback.format_exc()))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        conn.close()
