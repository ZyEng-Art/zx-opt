"""Open undirected ZX multigraphs. Wires have no orientation or layout.

Parallel wires, self loops, closed wire circles, degree-zero spiders and ordered
open ports are first-class. There is no extraction, size or candidate limit.
"""
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
import hashlib
import json

from .ring import C8, ZERO, ONE, INV_SQRT2, quarter, rational_phase
from .angles import record as phase_record, restore as restore_phase


@dataclass(frozen=True)
class Node:
    kind: str
    phase: Fraction = Fraction(0)
    allow_arbitrary: bool = field(default=False, compare=False, repr=False)

    def __post_init__(self):
        if self.kind not in ("Z", "X", "H", "B"):
            raise ValueError("Unknown ZX generator")
        normalize = rational_phase if self.allow_arbitrary else quarter
        object.__setattr__(self, "phase", normalize(self.phase))
        if self.kind in ("B", "H") and self.phase:
            raise ValueError("Boundary and H nodes have no phase")


@dataclass
class Diagram:
    nodes: dict = field(default_factory=dict)
    edges: list = field(default_factory=list)
    ports: tuple = ()
    loops: int = 0
    allow_arbitrary: bool = False

    def node(self, kind="Z", phase=0):
        v = max(self.nodes, default=-1) + 1
        self.nodes[v] = Node(kind, phase, self.allow_arbitrary)
        return v

    def edge(self, a, b):
        if a not in self.nodes or b not in self.nodes:
            raise ValueError("Wire endpoint absent")
        self.edges.append((a, b))
        return len(self.edges) - 1

    def boundary(self):
        b = self.node("B")
        self.ports += (b,)
        return b

    def leg(self, v):
        b = self.boundary()
        self.edge(v, b)
        return b

    def degree(self, v):
        return sum((a == v) + (b == v) for a, b in self.edges)

    def validate(self):
        if not isinstance(self.allow_arbitrary, bool):
            raise ValueError("allow_arbitrary must be boolean")
        if not isinstance(self.loops, int) or self.loops < 0:
            raise ValueError("Closed wire circle count must be nonnegative")
        if len(set(self.ports)) != len(self.ports):
            raise ValueError("Repeated open port")
        if set(self.ports) != {v for v, n in self.nodes.items() if n.kind == "B"}:
            raise ValueError("Ports must enumerate every boundary exactly once")
        for a, b in self.edges:
            if a not in self.nodes or b not in self.nodes:
                raise ValueError("Dangling wire")
        for v, n in self.nodes.items():
            if not self.allow_arbitrary:
                quarter(n.phase)
            if n.kind == "B" and self.degree(v) != 1:
                raise ValueError("Boundary must have exactly one wire")
            if n.kind == "H" and self.degree(v) != 2:
                raise ValueError("H must have exactly two wires")
        return self

    def clone(self):
        return Diagram(dict(self.nodes), list(self.edges), self.ports, self.loops, self.allow_arbitrary)

    def record(self):
        self.validate()
        row = dict(nodes={str(v): dict(kind=n.kind, phase=phase_record(n.phase))
                          for v, n in sorted(self.nodes.items())},
                   edges=[list(e) for e in self.edges], ports=list(self.ports), loops=self.loops)
        # Keep all historical Clifford+T records and SHA keys unchanged.
        if self.allow_arbitrary:
            row["allow_arbitrary"] = True
        return row

    @classmethod
    def from_record(cls, row):
        arbitrary = row.get("allow_arbitrary", False)
        return cls({int(v): Node(n["kind"], restore_phase(n["phase"]), arbitrary)
                    for v, n in row["nodes"].items()},
                   [tuple(e) for e in row["edges"]], tuple(row["ports"]),
                   row.get("loops", 0), arbitrary).validate()

    def digest(self):
        return hashlib.sha256(json.dumps(self.record(), sort_keys=True).encode()).hexdigest()

    def isomorphic(self, other):
        import networkx as nx
        def graph(d):
            g = nx.MultiGraph()
            for v, n in d.nodes.items():
                g.add_node(v, label=(n.kind, n.phase, d.ports.index(v) if v in d.ports else None))
            g.add_edges_from(d.edges)
            return g
        return (self.loops == other.loops and len(self.ports) == len(other.ports)
                and nx.is_isomorphic(graph(self), graph(other),
                                     node_match=lambda a, b: a["label"] == b["label"]))


def tensor_product(*diagrams):
    result = Diagram(allow_arbitrary=any(d.allow_arbitrary for d in diagrams))
    for d in diagrams:
        d.validate()
        mapping = {v: result.node(n.kind, n.phase) for v, n in d.nodes.items()}
        result.edges.extend((mapping[a], mapping[b]) for a, b in d.edges)
        result.ports += tuple(mapping[v] for v in d.ports)
        result.loops += d.loops
    return result


def join_ports(diagram, pairs):
    """Sew boundary pairs; trace and closed circles retain exact scalars."""
    d = diagram.clone().validate()
    used = [v for pair in pairs for v in pair]
    if len(used) != len(set(used)) or any(v not in d.ports for v in used):
        raise ValueError("Sewing needs disjoint boundary pairs")
    for a, b in pairs:
        d.edge(a, b)
    d.ports = tuple(v for v in d.ports if v not in used)
    return suppress(d, set(used))


def suppress(diagram, vertices):
    """Erase specified degree-two wire junctions, including all-junction circles."""
    d = diagram.clone()
    for v in sorted(vertices):
        incidences = [(i, s) for i, e in enumerate(d.edges) for s in (0, 1) if e[s] == v]
        if len(incidences) != 2:
            raise ValueError("A wire junction must have degree two")
        indices = {i for i, _ in incidences}
        if len(indices) == 1:
            d.loops += 1
        else:
            ends = [d.edges[i][1-s] for i, s in incidences]
            d.edges.append(tuple(ends))
        d.edges = [e for i, e in enumerate(d.edges) if i not in indices]
        del d.nodes[v]
    return d.validate()


def z_network(diagram):
    """Eliminate explicit H chains, then colour-change X to Z exactly.

    Returns (vertices, normalized edges (a,b,H-parity), scalar). This is also
    useful for exporting a generic diagram without dropping parallel wires.
    """
    d = diagram.validate()
    vertices = {v: n for v, n in d.nodes.items() if n.kind != "H"}
    incidences = {v: [] for v in d.nodes}
    for i, e in enumerate(d.edges):
        for s, v in enumerate(e):
            incidences[v].append((i, s))
    used = set()
    edges = []
    scalar = C8.of(2 ** d.loops)
    for start in vertices:
        for i, s in incidences[start]:
            if i in used:
                continue
            parity = int(d.nodes[start].kind == "X")
            edge, side = i, s
            while True:
                used.add(edge)
                end = d.edges[edge][1-side]
                if end in vertices:
                    parity ^= int(d.nodes[end].kind == "X")
                    edges.append((start, end, parity))
                    break
                parity ^= 1
                options = [h for h in incidences[end] if h != (edge, 1-side)]
                edge, side = options[0]
    # Remaining components consist entirely of degree-two H nodes.
    for i in range(len(d.edges)):
        if i in used:
            continue
        edge, side, parity = i, 0, 0
        while edge not in used:
            used.add(edge)
            end = d.edges[edge][1-side]
            parity ^= 1
            edge, side = next(h for h in incidences[end] if h != (edge, 1-side))
        scalar *= 0 if parity else 2
    return vertices, edges, scalar


def exact_tensor(diagram):
    """Independent exact variable elimination; entries in ordered port basis.

    No floating arithmetic, amplitude normalization, or global-phase alignment.
    Resource limits belong to callers, not to the definition of a legal rewrite.
    """
    d = diagram.validate()
    # General-angle contexts use the independently checked numeric operator
    # path. Never truncate a non-Clifford+T phase into the C8 exact ring.
    for n in d.nodes.values():
        quarter(n.phase)
    vertices, edges, scalar = z_network(d)
    factors = []
    for v, n in vertices.items():
        if n.kind != "B":
            factors.append(((v,), {(0,): ONE, (1,): C8.phase(int(n.phase * 4))}))
    for a, b, parity in edges:
        scope = tuple(sorted({a, b}))
        table = {}
        for values in product((0, 1), repeat=len(scope)):
            bits = dict(zip(scope, values))
            x, y = bits[a], bits[b]
            table[values] = (INV_SQRT2 * (-1 if x and y else 1)) if parity else C8.of(x == y)
        factors.append((scope, table))
    internal = set(vertices) - set(d.ports)
    while internal:
        def scope_for(v):
            return tuple(sorted({w for s, _ in factors if v in s for w in s}))
        v = min(internal, key=lambda x: (len(scope_for(x)), x))
        scope = scope_for(v)
        joined = [f for f in factors if v in f[0]]
        factors = [f for f in factors if v not in f[0]]
        remaining = tuple(w for w in scope if w != v)
        table = {}
        for values in product((0, 1), repeat=len(remaining)):
            bits = dict(zip(remaining, values))
            total = ZERO
            for bit in (0, 1):
                bits[v] = bit
                value = ONE
                for s, t in joined:
                    value *= t[tuple(bits[w] for w in s)]
                    if not value:
                        break
                total += value
            table[values] = total
        factors.append((remaining, table))
        internal.remove(v)
    result = []
    for values in product((0, 1), repeat=len(d.ports)):
        bits = dict(zip(d.ports, values))
        value = scalar
        for s, t in factors:
            value *= t[tuple(bits[w] for w in s)]
        result.append(value)
    return tuple(result)
