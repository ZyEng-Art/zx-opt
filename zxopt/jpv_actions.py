"""Bounded proposals from the complete kernel, with checked exact witnesses.

Proposal bounds are search resources, not a change to the underlying calculus.
S/I/H use incidence-based proposals to avoid factorial enumeration; other rules
match topology once and infer the finite phase parameters, rather than running
512 independent C matchers per state.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from functools import lru_cache
import hashlib
from itertools import product
import json
import time

import numpy as np

from .complete import Diagram, axiom, actions as rule_actions, exact_tensor
from .complete.rewrite import Match, RuleAction, matches, replace, MatchLimit
from .complete.ring import is_quarter


def connected_ports(graph):
    adjacent = {v: [] for v in graph.nodes}
    for a, b in graph.edges:
        adjacent[a].append(b); adjacent[b].append(a)
    reached = set(graph.ports); pending = list(reached)
    while pending:
        v = pending.pop()
        for w in adjacent[v]:
            if w not in reached:
                reached.add(w); pending.append(w)
    return reached


def incidences(graph):
    result = {v: [] for v in graph.nodes}
    for i, edge in enumerate(graph.edges):
        for s, v in enumerate(edge):
            result[v].append((i, s))
    return result


def single_match(source, host_vertex, halves):
    internal = [v for v, n in source.nodes.items() if n.kind != "B"]
    if len(internal) != 1 or len(halves) != len(source.ports):
        raise ValueError("Not a one-spider pattern")
    return Match(((internal[0], host_vertex),), (),
                 tuple((p, e, s) for p, (e, s) in zip(source.ports, halves)), ())


@lru_cache(maxsize=1)
def hh_identity():
    """Prove H-H = wire by JPV I_X backward, H forward, I_Z forward."""
    source = Diagram(); a, b = source.node("H"), source.node("H")
    source.edge(a, b); source.leg(a); source.leg(b)
    target = Diagram(); a, b = target.boundary(), target.boundary(); target.edge(a, b)
    g = source; proof = []
    insert = next(a for a in rule_actions(g, axiom("I", color="X"), "backward")
                  if a.match.wire_edges[0][1] == 0)
    for chosen in (insert,):
        g = chosen.apply(g); proof.append(chosen.record())
    chosen = next(rule_actions(g, axiom("H", alpha=F(0), arity=2)))
    g = chosen.apply(g); proof.append(chosen.record())
    chosen = next(rule_actions(g, axiom("I")))
    g = chosen.apply(g); proof.append(chosen.record())
    assert g.isomorphic(target) and exact_tensor(source) == exact_tensor(target)
    receipt = dict(source=source.record(), target=target.record(), proof=proof)
    digest = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
    return source, target, receipt, digest


def cancel_h_pairs(graph):
    """Only the proven HH identity; do not fuse/simplify away chosen JPV steps."""
    source, target, _, proof_sha = hh_identity()
    g = graph; receipts = []
    while True:
        found = next(((i, a, b) for i, (a, b) in enumerate(g.edges)
                      if a != b and g.nodes[a].kind == g.nodes[b].kind == "H"), None)
        if found is None:
            return g, receipts
        e, a, b = found
        hs = incidences(g)
        ah = next(h for h in hs[a] if h[0] != e)
        bh = next(h for h in hs[b] if h[0] != e)
        m = Match(((0, a), (1, b)), ((0, e),),
                  ((source.ports[0], *ah), (source.ports[1], *bh)), ())
        child = replace(g, source, target, m)
        receipts.append(dict(derived="HH", derivation_sha256=proof_sha,
                             parent_sha256=g.digest(), match=m.record(), child_sha256=child.digest()))
        g = child


def replay_h_pairs(graph, receipts):
    source, target, _, sha = hh_identity()
    g = graph
    for row in receipts:
        if row["derived"] != "HH" or row["derivation_sha256"] != sha or row["parent_sha256"] != g.digest():
            raise ValueError("Invalid HH derivation receipt")
        g = replace(g, source, target, Match.from_record(row["match"]))
        if g.digest() != row["child_sha256"]:
            raise ValueError("HH child digest mismatch")
    return g


@lru_cache(maxsize=128)
def compiled(name, direction, color, arity=0):
    params = {k: F(0) for k in dict(K=("alpha",), SUP=("alpha",), C=("alpha", "beta", "gamma"),
                                  H=("alpha",)).get(name, ())}
    if name == "H":
        params["arity"] = arity
    base = axiom(name, color=color, **params)
    source = base.left if direction == "forward" else base.right
    internal = tuple(v for v in source.nodes if v not in source.ports)
    phase_names = tuple(k for k in ("alpha", "beta", "gamma") if k in params)
    constant = [int(source.nodes[v].phase * 4) for v in internal]
    coefficients = []
    for k in phase_names:
        probe = dict(params); probe[k] = F(1, 4)
        r = axiom(name, color=color, **probe)
        s = r.left if direction == "forward" else r.right
        coefficients.append([(int(s.nodes[v].phase*4)-c) % 8 for v, c in zip(internal, constant)])
    lookup = {}
    for values in product(range(8), repeat=len(phase_names)):
        signature = tuple((c + sum(a[j] * q for a, q in zip(coefficients, values))) % 8
                          for j, c in enumerate(constant))
        lookup.setdefault(signature, []).append(dict(zip(phase_names, (F(v, 4) for v in values))))
    return source, internal, lookup, {k: v for k, v in params.items() if k not in phase_names}


class Proposals:
    def __init__(self, seed=11, cap=32, matching_work=2048):
        self.seed = seed; self.cap = cap; self.matching_work = matching_work
        self.rounds = {}; self.stats = dict(calls=0, matching_work_exhausted=0, phase_inferred=0,
                                           non_clifford_matches_skipped=0)
        # Compilation depends on the calculus, not on observations or rewards.
        for name in ("CP", "B", "K", "EU", "SUP", "E", "C", "BW"):
            for direction, color in product(("forward", "backward"), ("Z", "X")):
                compiled(name, direction, color)
        hh_identity()

    def build(self, graph):
        if not self.cap:
            return []
        key = graph.digest(); visit = self.rounds.get(key, 0); self.rounds[key] = visit+1
        digest = hashlib.sha256((str(self.seed)+key+":"+str(visit)).encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        hs = incidences(graph); reached = connected_ports(graph)
        # Random order within the boundary-connected and scalar groups, fixed
        # by explicit state/visit RNG; never affected by PPO's torch RNG.
        vertices = list(graph.nodes); rng.shuffle(vertices)
        vertices.sort(key=lambda v: v not in reached)
        host = graph.clone(); host.nodes = {v: graph.nodes[v] for v in vertices}
        parent = key
        buckets = {}
        per = max(2, (self.cap+10)//11)

        def add(rule, direction, match):
            family = rule.name + ":" + direction
            if len(buckets.get(family, ())) < per:
                buckets.setdefault(family, []).append(RuleAction(rule, direction, match, parent))

        # Spider fusion: all direct same-color neighbors, including parallel wires.
        pairs = {}
        for i, (a, b) in enumerate(graph.edges):
            if a != b and graph.nodes[a].kind == graph.nodes[b].kind and graph.nodes[a].kind in ("Z", "X"):
                pairs.setdefault(tuple(sorted((a, b))), []).append(i)
        pair_items = [(pair, es) for pair, es in pairs.items()
                      if all(is_quarter(graph.nodes[v].phase) for v in pair)]
        rng.shuffle(pair_items)
        pair_items.sort(key=lambda item: not any(v in reached for v in item[0]))
        for (a, b), es in pair_items[:per]:
            rule = axiom("S", color=graph.nodes[a].kind, alpha=graph.nodes[a].phase,
                         beta=graph.nodes[b].phase, left_legs=len(hs[a])-len(es),
                         right_legs=len(hs[b])-len(es), links=len(es))
            bindings = []
            for p, v in zip(rule.left.ports, [a]*(len(hs[a])-len(es)) + [b]*(len(hs[b])-len(es))):
                available = [h for h in hs[v] if h[0] not in es and h not in [x[1:] for x in bindings]]
                bindings.append((p, *available[0]))
            add(rule, "forward", Match(((0, a), (1, b)), tuple(zip(range(len(es)), es)), tuple(bindings), ()))

        for v in vertices:
            n = graph.nodes[v]
            if n.kind not in ("Z", "X"):
                continue
            if not is_quarter(n.phase):
                # This finite JPV menu only instantiates Clifford+T templates.
                # Keep other angles unchanged as the surrounding context.
                continue
            shuffled = list(hs[v]); rng.shuffle(shuffled)
            if len(buckets.get("S:backward", ())) < per:
                count = int(rng.integers(len(shuffled)+1)); alpha = F(int(rng.integers(8)), 4)
                rule = axiom("S", color=n.kind, alpha=alpha, beta=n.phase-alpha,
                             left_legs=count, right_legs=len(shuffled)-count)
                add(rule, "backward", single_match(rule.right, v, shuffled))
            if not n.phase and len(shuffled) == 2 and len(buckets.get("I:forward", ())) < per:
                rule = axiom("I", color=n.kind)
                add(rule, "forward", single_match(rule.left, v, shuffled))
            if len(buckets.get("H:backward", ())) < per:
                rule = axiom("H", color=n.kind, alpha=n.phase, arity=len(shuffled))
                add(rule, "backward", single_match(rule.right, v, shuffled))

        edge_order = list(range(len(graph.edges))); rng.shuffle(edge_order)
        edge_order.sort(key=lambda e: not any(v in reached for v in graph.edges[e]))
        for i in edge_order[:per]:
            rule = axiom("I", color="Z" if rng.integers(2) else "X")
            add(rule, "backward", Match(wire_edges=((0, i, int(rng.integers(2))),)))

        # Fixed branch-count bounds preserve paired prefixes across machine load.
        jobs = [(name, direction, color, 0) for name in ("CP", "B", "K", "EU", "SUP", "E", "C", "BW")
                for direction, color in product(("forward", "backward"), ("Z", "X"))]
        arities = sorted({len(hs[v]) for v in vertices if graph.nodes[v].kind in ("Z", "X")})
        jobs += [("H", "forward", color, arity) for color in ("Z", "X") for arity in arities]
        rng.shuffle(jobs)
        for name, direction, color, arity in jobs:
            if len(buckets.get(name+":"+direction, ())) >= per:
                continue
            try:
                pattern, internal, lookup, fixed = compiled(name, direction, color, arity)
                attempts = 0
                for match in matches(host, pattern, ignore_phases=True, work_limit=self.matching_work):
                    attempts += 1
                    mapping = dict(match.vertices)
                    phases = tuple(graph.nodes[mapping[v]].phase for v in internal)
                    if not all(is_quarter(p) for p in phases):
                        self.stats["non_clifford_matches_skipped"] += 1
                        if attempts >= 64:
                            break
                        continue
                    signature = tuple(int(p*4) for p in phases)
                    for params in lookup.get(signature, ()):
                        rule = axiom(name, color=color, **fixed, **params)
                        add(rule, direction, match); self.stats["phase_inferred"] += 1
                        if len(buckets.get(name+":"+direction, ())) >= per:
                            break
                    if len(buckets.get(name+":"+direction, ())) >= per or attempts >= 64:
                        break
            except MatchLimit:
                self.stats["matching_work_exhausted"] += 1
        result = []; signatures = set()
        # Rotate the family order every visit. A bucket with many scalar matches
        # cannot consume the entire global proposal cap.
        order = sorted(buckets); rng.shuffle(order)
        for offset in range(per):
            for family in order:
                if len(buckets[family]) <= offset:
                    continue
                action = buckets[family][offset]
                signature = json.dumps(action.record(), sort_keys=True)
                if signature not in signatures:
                    signatures.add(signature); result.append(action)
                if len(result) >= self.cap:
                    self.stats["calls"] += 1
                    return result
        self.stats["calls"] += 1
        return result
