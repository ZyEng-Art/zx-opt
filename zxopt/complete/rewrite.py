"""Bidirectional open-subdiagram replacement, including arbitrary contexts.

Matches distinguish wire incidences, so a context may join two ports back to
one another, attach them to the same outside spider, or contain closed wires.
No semantic equality oracle is used to admit an action: its witness is a named
JPV axiom instance plus an explicitly checked graph embedding.
"""
from dataclasses import dataclass
from itertools import permutations

from .axioms import Axiom, instances_at
from .diagram import Diagram, suppress


@dataclass(frozen=True)
class Match:
    vertices: tuple = ()           # (pattern internal vertex, host vertex)
    internal_edges: tuple = ()     # (pattern internal edge, host edge)
    port_halves: tuple = ()        # (pattern boundary, host edge, inside side)
    wire_edges: tuple = ()         # (boundary-boundary edge, host edge, reversal)
                                  # negative host edge means closed wire circle

    def record(self):
        return {k: [list(v) for v in getattr(self, k)] for k in
                ("vertices", "internal_edges", "port_halves", "wire_edges")}

    @classmethod
    def from_record(cls, row):
        return cls(*(tuple(tuple(v) for v in row[k]) for k in
                     ("vertices", "internal_edges", "port_halves", "wire_edges")))


def _parts(pattern):
    internal = set(pattern.nodes) - set(pattern.ports)
    ii, ip, pp = [], [], []
    for i, (a, b) in enumerate(pattern.edges):
        if a in internal and b in internal:
            ii.append(i)
        elif a in internal or b in internal:
            ip.append((a if a in pattern.ports else b, b if a in pattern.ports else a))
        else:
            pp.append(i)
    return internal, ii, ip, pp


class MatchLimit(RuntimeError):
    """Optional caller-selected deterministic matching work budget exhausted."""


def matches(host, pattern, ignore_phases=False, work_limit=None):
    """Enumerate every finite embedding of an instantiated registry pattern.

    A concrete pattern has finitely many matches in a finite host. In contrast,
    the axiom schema has unbounded arity; use actions_at(rank) for fair stages.
    """
    host.validate(); pattern.validate()
    work = 0
    def tick():
        nonlocal work
        work += 1
        if work_limit is not None and work > work_limit:
            raise MatchLimit("Pattern matching work budget exhausted")
    if host.loops < pattern.loops:
        return
    internal, ii, ip, pp = _parts(pattern)
    if pp and (internal or len(pp) != 1):
        raise ValueError("Mixed/free multiwire custom patterns need further wire subdivision; not a JPV registry pattern")
    hd = {v: 0 for v in host.nodes}
    pd = {v: 0 for v in pattern.nodes}
    for a, b in host.edges:
        hd[a] += 1; hd[b] += 1
    for a, b in pattern.edges:
        pd[a] += 1; pd[b] += 1
    host_edges = {}
    pattern_edges = {}
    for i, edge in enumerate(host.edges):
        host_edges.setdefault(tuple(sorted(edge)), []).append(i)
    for i in ii:
        pair = tuple(sorted(pattern.edges[i])); pattern_edges[pair] = pattern_edges.get(pair, 0)+1
    labels = {}
    for w, n in host.nodes.items():
        if n.kind != "B":
            labels.setdefault((n.kind, None if ignore_phases else n.phase, hd[w]), []).append(w)
    choices = {v: labels.get((pattern.nodes[v].kind,
                             None if ignore_phases else pattern.nodes[v].phase, pd[v]), []) for v in internal}
    order = sorted(internal, key=lambda v: (len(choices[v]), -pd[v], v))
    mapping = {}

    def vertex_assign(k):
        if k == len(order):
            yield from edge_assign(0, {}, set())
            return
        v = order[k]
        for w in choices[v]:
            tick()
            if w in mapping.values():
                continue
            ok = True
            for u, target in mapping.items():
                needed = pattern_edges.get(tuple(sorted((u, v))), 0)
                actual = len(host_edges.get(tuple(sorted((target, w))), ()))
                if actual < needed:
                    ok = False; break
            if ok:
                mapping[v] = w
                yield from vertex_assign(k+1)
                del mapping[v]

    def edge_assign(k, selected, used):
        if k == len(ii):
            groups = {}
            for port, v in ip:
                groups.setdefault(v, []).append(port)
            yield from port_assign(list(groups.items()), 0, selected, used, {})
            return
        edge = ii[k]
        a, b = pattern.edges[edge]
        for j in host_edges.get(tuple(sorted((mapping[a], mapping[b]))), ()):
            tick()
            if j not in used:
                selected[edge] = j
                yield from edge_assign(k+1, selected, used | {j})
                del selected[edge]

    def port_assign(groups, k, selected, used, bindings):
        if k == len(groups):
            excluded = used | {e for e, _ in bindings.values()}
            yield from wire_assign(0, selected, bindings, {}, excluded)
            return
        v, ports = groups[k]
        halves = [(i, s) for i, edge in enumerate(host.edges) if i not in used
                  for s in (0, 1) if edge[s] == mapping[v]]
        if len(halves) != len(ports):
            return
        for assignment in permutations(halves):
            tick()
            bindings.update(zip(ports, assignment))
            yield from port_assign(groups, k+1, selected, used, bindings)
            for port in ports:
                del bindings[port]

    def wire_assign(k, selected, bindings, wires, excluded):
        if k == len(pp):
            yield Match(tuple(sorted(mapping.items())), tuple(sorted(selected.items())),
                        tuple(sorted((p, e, s) for p, (e, s) in bindings.items())),
                        tuple(sorted((p, e, s) for p, (e, s) in wires.items())))
            return
        p = pp[k]
        for i, (a, b) in enumerate(host.edges):
            tick()
            if i in excluded or a in mapping.values() or b in mapping.values():
                continue
            for reversal in (0, 1):
                wires[p] = (i, reversal)
                yield from wire_assign(k+1, selected, bindings, wires, excluded | {i})
        for loop in range(pattern.loops, host.loops):
            edge = -loop-1
            if edge not in excluded:
                wires[p] = (edge, 0)
                yield from wire_assign(k+1, selected, bindings, wires, excluded | {edge})
        wires.pop(p, None)

    yield from vertex_assign(0)


def _validate_match(host, pattern, match):
    host.validate(); pattern.validate()
    internal, ii, ip, pp = _parts(pattern)
    vm, em = dict(match.vertices), dict(match.internal_edges)
    ph = {p: (e, s) for p, e, s in match.port_halves}
    we = {p: (e, s) for p, e, s in match.wire_edges}
    if (len(vm) != len(match.vertices) or set(vm) != internal
            or len(set(vm.values())) != len(vm) or len(em) != len(match.internal_edges)
            or set(em) != set(ii) or len(set(em.values())) != len(em)
            or len(ph) != len(match.port_halves) or set(ph) != {p for p, _ in ip}
            or len(we) != len(match.wire_edges) or set(we) != set(pp)):
        raise ValueError("Incomplete or noninjective occurrence")
    for v, w in vm.items():
        if (w not in host.nodes or pattern.nodes[v] != host.nodes[w]
                or host.degree(w) != pattern.degree(v)):
            raise ValueError("Vertex label or degree mismatch")
    for i, j in em.items():
        if (not 0 <= j < len(host.edges)
                or sorted(host.edges[j]) != sorted(vm[v] for v in pattern.edges[i])):
            raise ValueError("Internal wire mismatch")
    occupied = {(e, s) for e in em.values() for s in (0, 1)}
    for p, v in ip:
        e, s = ph[p]
        if (not 0 <= e < len(host.edges) or s not in (0, 1)
                or host.edges[e][s] != vm[v] or (e, s) in occupied):
            raise ValueError("Cut wire incidence mismatch")
        occupied.add((e, s))
    expected = {(i, s) for i, e in enumerate(host.edges) for s in (0, 1) if e[s] in vm.values()}
    if occupied != expected:
        raise ValueError("Occurrence leaves an unaccounted incident wire")
    used = set(em.values()) | {e for e, _ in ph.values()}
    for e, s in we.values():
        if s not in (0, 1) or e in used:
            raise ValueError("Repeated or misoriented wire component")
        if e < 0:
            if not pattern.loops <= -e-1 < host.loops or s != 0:
                raise ValueError("Closed circle occurrence absent")
        elif (e >= len(host.edges) or any(v in vm.values() for v in host.edges[e])):
            raise ValueError("Wire component intersects pattern interior")
        used.add(e)
    if host.loops < pattern.loops:
        raise ValueError("Closed circles absent")
    return vm, em, ph, we, used


def replace(host, source, target, match):
    """Apply an already chosen local graph equation in its full wire context."""
    if len(source.ports) != len(target.ports):
        raise ValueError("An equation must preserve its ordered interface")
    target.validate()
    vm, em, ph, we, used = _validate_match(host, source, match)
    selected = set(vm.values())
    result = Diagram({v: n for v, n in host.nodes.items() if v not in selected},
                     [e for i, e in enumerate(host.edges) if i not in used], host.ports,
                     host.loops - source.loops - sum(e < 0 for e, _ in we.values()) + target.loops,
                     host.allow_arbitrary or source.allow_arbitrary or target.allow_arbitrary)
    junction = {p: result.node("B") for p in source.ports}
    inverse_halves = {h: p for p, h in ph.items()}
    for p, (e, s) in ph.items():
        end = host.edges[e][1-s]
        if end not in selected:
            result.edge(junction[p], end)
        else:
            q = inverse_halves[(e, 1-s)]
            if p < q:
                result.edge(junction[p], junction[q])
    for pe, (he, reversal) in we.items():
        p, q = source.edges[pe]
        if he < 0:
            result.edge(junction[p], junction[q])
        else:
            a, b = host.edges[he]
            if reversal:
                a, b = b, a
            result.edge(junction[p], a); result.edge(junction[q], b)
    tm = {v: result.node(n.kind, n.phase) for v, n in target.nodes.items() if n.kind != "B"}
    tm.update({q: junction[p] for p, q in zip(source.ports, target.ports)})
    for a, b in target.edges:
        result.edge(tm[a], tm[b])
    return suppress(result, set(junction.values()))


@dataclass(frozen=True)
class RuleAction:
    axiom: Axiom
    direction: str
    match: Match
    parent_sha256: str

    def record(self):
        return dict(axiom=self.axiom.record(), direction=self.direction,
                    match=self.match.record(), parent_sha256=self.parent_sha256)

    @classmethod
    def from_record(cls, row):
        return cls(Axiom.from_record(row["axiom"]), row["direction"],
                   Match.from_record(row["match"]), row["parent_sha256"])

    def apply(self, host):
        if host.digest() != self.parent_sha256:
            raise ValueError("Action belongs to a different graph representation")
        if self.direction not in ("forward", "backward"):
            raise ValueError("Direction must be forward/backward")
        # Reconstruct the trusted named template, rather than trusting mutable
        # pattern objects received from a caller or serialized proof log.
        rule = Axiom.from_record(self.axiom.record())
        l, r = rule.left, rule.right
        return replace(host, l, r, self.match) if self.direction == "forward" else replace(host, r, l, self.match)


def actions(host, rule, direction="forward"):
    if direction not in ("forward", "backward"):
        raise ValueError("Direction must be forward/backward")
    source = rule.left if direction == "forward" else rule.right
    digest = host.digest()
    for match in matches(host, source):
        yield RuleAction(rule, direction, match, digest)


def actions_at(host, rank):
    for rule in instances_at(rank):
        yield from actions(host, rule, "forward")
        yield from actions(host, rule, "backward")
