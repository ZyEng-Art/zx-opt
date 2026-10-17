# SPDX-License-Identifier: Apache-2.0
# Match-update logic follows PyZX (C)2018 Aleks Kissinger and John van de Wetering.
"""Same CFlow choices, with sparse match updates and exact integer-bitset scores.

The upstream flow driver, match enumeration, ordering and acceptance test remain.
Bitsets are rebuilt for each newly scored graph, never reused across rewrites.
"""
import types
from collections import defaultdict

import pyzx.heuristics as heuristics
import pyzx.simplify as simplify
import pyzx.rules as rules
from pivot_candidates import match_pivot_unfuse
try:
    from _native_scores import score_batch as native_score_batch
except ImportError:
    native_score_batch = None

POPCOUNT = int.bit_count if hasattr(int, 'bit_count') else lambda value: bin(value).count('1')


def vertices(bits):
    while bits:
        low = bits & -bits
        yield low.bit_length() - 1
        bits ^= low


class ExactScores:
    def __init__(self, verify=False, mode='sets'):
        self.graph = None
        self.adjacency = None
        self.verify = verify
        assert mode in ['sets', 'bits']
        self.mode = mode
        self.neighbor_sets = self.pairs = None
        self.statistics = defaultdict(int)

    def prepare(self, graph):
        self.graph = graph
        ids = list(graph.vertices())
        self.pairs = {}
        if self.mode == 'sets':
            self.neighbor_sets = {v: set(graph.neighbors(v)) for v in ids}
            self.statistics['graph_rebuilds'] += 1
            return
        if not all(isinstance(v, int) and 0 <= v < 100000 for v in ids):
            self.adjacency = None
            return
        self.adjacency = {v: sum(1 << n for n in graph.neighbors(v)) for v in ids}
        self.statistics['graph_rebuilds'] += 1

    def pivot(self, graph, v0, v1, unfusions=((), ())):
        if graph is not self.graph:
            self.prepare(graph)
        if self.mode == 'sets':
            pair = v0, v1
            neighbors = self.neighbor_sets
            if pair not in self.pairs:
                base0, base1 = neighbors[v0] - {v1}, neighbors[v1] - {v0}
                shared = base0 & base1
                deg0 = {v: len(neighbors[v] & base1) for v in base0}
                deg1 = {v: len(neighbors[v] & base0) for v in base1}
                degs = {v: len(neighbors[v] & shared) for v in shared}
                self.pairs[pair] = (base0, base1, shared, deg0, deg1, degs,
                                    sum(deg0.values()), sum(degs.values()) // 2, {}, {})
            base0, base1, shared, deg0, deg1, degs, cross, internal, cache0, cache1 = self.pairs[pair]
            u0, u1 = unfusions
            if u0 not in cache0:
                removed = tuple(v for v in u0 if v in base0)
                cache0[u0] = (removed, frozenset(removed) & shared, sum(deg0[v] for v in removed))
            if u1 not in cache1:
                removed = tuple(v for v in u1 if v in base1)
                cache1[u1] = (removed, frozenset(removed) & shared, sum(deg1[v] for v in removed))
            r0, rs0, delta0 = cache0[u0]
            r1, rs1, delta1 = cache1[u1]
            removed_shared = rs0 | rs1
            ns = len(shared) - len(removed_shared)
            na = len(base0) - len(r0) - ns + bool(u0)
            nb = len(base1) - len(r1) - ns + bool(u1)
            remaining_cross = cross - delta0 - delta1 + sum(v in neighbors[u] for u in r0 for v in r1)
            remaining_internal = 0
            if shared:
                remaining_internal = internal - sum(degs[v] for v in removed_shared)
                remaining_internal += sum(u < v and v in neighbors[u] for u in removed_shared for v in removed_shared)
            edges = remaining_cross - 2 * remaining_internal
            unfused = bool(u0) + bool(u1)
            result = (2 * edges - (na * nb + na * ns + nb * ns) + na + 2 * ns + nb + 1 - 2 * unfused,
                      2 - 2 * unfused)
            self.statistics['pivot_scores'] += 1
            if self.verify:
                assert result == heuristics.pivot_statistics(graph, v0, v1, unfusions)
                self.statistics['pivot_scores_verified'] += 1
            return result
        if self.adjacency is None:
            self.statistics['fallbacks'] += 1
            return heuristics.pivot_statistics(graph, v0, v1, unfusions)
        u0 = u1 = 0
        for v in unfusions[0]:
            u0 |= 1 << v
        for v in unfusions[1]:
            u1 |= 1 << v
        a = self.adjacency[v0] & ~((1 << v1) | u0)
        b = self.adjacency[v1] & ~((1 << v0) | u1)
        shared = a & b
        a &= ~shared
        b &= ~shared
        na, nb, ns = POPCOUNT(a) + bool(u0), POPCOUNT(b) + bool(u1), POPCOUNT(shared)
        other = b | shared
        edges = sum(POPCOUNT(self.adjacency[v] & other) for v in vertices(a))
        edges += sum(POPCOUNT(self.adjacency[v] & shared) for v in vertices(b))
        unfused = bool(u0) + bool(u1)
        result = (2 * edges - (na * nb + na * ns + nb * ns) + na + 2 * ns + nb + 1 - 2 * unfused,
                  2 - 2 * unfused)
        self.statistics['pivot_scores'] += 1
        if self.verify:
            assert result == heuristics.pivot_statistics(graph, v0, v1, unfusions)
            self.statistics['pivot_scores_verified'] += 1
        return result

    def score(self, g, match, score_weights=(1, 1, 1), **kwargs):
        if match[0]:
            if g is not self.graph:
                self.prepare(g)
            if self.mode != 'sets':
                return simplify.match_score_2Q_simp(g, match, score_weights, **kwargs)
            vertex, neighbors, unfusions = match[0]
            remaining = set(neighbors) - set(unfusions)
            count = len(remaining) + bool(unfusions)
            twice_edges = sum(len(self.neighbor_sets[v] & remaining) for v in remaining)
            removed = 1 - 2 * bool(unfusions)
            improvement = twice_edges - count * (count - 1) // 2 + count - 1
            result = score_weights[0] * improvement if improvement > 0 or (improvement == 0 and removed > 0) else None
            self.statistics['lcomp_scores'] += 1
            if self.verify:
                assert result == simplify.match_score_2Q_simp(g, match, score_weights, **kwargs)
                self.statistics['lcomp_scores_verified'] += 1
            return result
        if not match[1]:
            return simplify.match_score_2Q_simp(g, match, score_weights, **kwargs)
        edges, removed = self.pivot(g, match[1][0], match[1][1], match[1][2])
        improvement = edges - removed
        if improvement > 0 or (improvement == 0 and removed > 0):
            return score_weights[1] * improvement
        return None


def update_matches(g_before, g_after, current_matches, get_matches, match_score,
                   matchf=None, **kwargs):
    before = set(g_before.vertices())
    after = set(g_after.vertices())
    removed = before - after
    touched = set(g_after.vertices_to_update)
    changed = {v for v in after if v in touched or
               g_after.neighbors(v) != g_before.graph.get(v, {}).keys()}
    affected = changed | {n for v in changed for n in g_after.neighbors(v)}
    # Same edge set as the quadratic cross product, visited via adjacency.
    edges = {(v, n) for v in affected for n in g_after.neighbors(v)
             if v < n and n in affected}
    candidates = affected | edges
    filter_match = lambda candidate: candidate in candidates and (not matchf or matchf(candidate))
    new_matches = get_matches(g_after, filter_match, **kwargs)
    invalid = affected | removed
    updated = {}
    for match, score in current_matches.items():
        if match[0] and match[0][0] in invalid:
            continue
        if match[1] and (all(v in affected for v in [match[1][0], match[1][1]]) or
                         any(v in removed for v in [match[1][0], match[1][1]])):
            continue
        if match[2] and match[2][0] in invalid:
            continue
        updated[match] = score
    for match in new_matches:
        score = match_score(g=g_after, match=match, **kwargs)
        if score is not None:
            updated[match] = score
    g_after.vertices_to_update = []
    return updated


def flow(graph, verify=False, statistics=None, score_mode='sets', native=True, filter_nonimproving=True, identity_lookup=False, **kwargs):
    scores = ExactScores(verify, mode=score_mode)
    counts = defaultdict(int)

    def update(*args, **options):
        if verify:
            before, after, matches, get_matches, scorer = args[:5]
            shadow = after.clone()
            expected = simplify.update_2Q_simp_matches(before, shadow, matches, get_matches,
                                                       simplify.match_score_2Q_simp, *args[5:], **options)
        actual = update_matches(*args, **options)
        if verify:
            # Dict insertion order determines tie-breaking by max().
            assert list(expected.items()) == list(actual.items())
            counts['match_updates_verified'] += 1
        counts['match_updates'] += 1
        return actual

    match_namespace = dict(rules.match_2Q_simp.__globals__)
    match_namespace['match_pivot_unfuse'] = match_pivot_unfuse
    matcher = types.FunctionType(rules.match_2Q_simp.__code__, match_namespace, rules.match_2Q_simp.__name__, rules.match_2Q_simp.__defaults__)
    score_function = scores.score
    if native and score_mode == 'sets' and native_score_batch is not None:
        original_matcher = matcher
        score_lookup = {}
        def matcher(g, *args, **options):
            matches = original_matcher(g, *args, **options)
            weights = options.get('score_weights', (1, 1, 1))
            try:
                values = native_score_batch(g.graph, matches, weights)
                counts['native_batches'] += 1
                counts['native_scores'] += len(matches)
            except (TypeError, OverflowError):
                values = [scores.score(g, match, weights) for match in matches]
                counts['native_fallback_batches'] += 1
            if verify:
                assert values == [simplify.match_score_2Q_simp(g, match, weights) for match in matches]
                counts['native_scores_verified'] += len(matches)
            pairs = [(match, value) for match, value in zip(matches, values)
                     if not filter_nonimproving or value is not None]
            counts['discarded_pair_allocations_avoided'] += len(matches) - len(pairs)
            score_lookup.clear()
            if identity_lookup:
                score_lookup.update((id(match), value) for match, value in pairs)
            else:
                score_lookup.update(pairs)
            return [match for match, value in pairs]
        def score_function(g, match, score_weights=(1, 1, 1), **options):
            return score_lookup[id(match) if identity_lookup else match]
    namespace = dict(simplify.flow_2Q_simp.__globals__)
    namespace['match_2Q_simp'] = matcher
    namespace.update(match_score_2Q_simp=score_function, update_2Q_simp_matches=update)
    driver = types.FunctionType(simplify.flow_2Q_simp.__code__, namespace,
                                simplify.flow_2Q_simp.__name__, simplify.flow_2Q_simp.__defaults__)
    result = driver(graph, **kwargs)
    if statistics is not None:
        for key, value in dict(scores.statistics, **counts).items():
            statistics[key] = statistics.get(key, 0) + value
    return result
