"""Finite, controlled expansion and phase-polynomial rewrites.

All phases are exact rational multiples of pi. A split uses an H-H identity
bridge, retaining a graph-like diagram. No general flow-preservation claim:
the caller must successfully extract and verify before admitting a candidate.
"""
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations

from pyzx.utils import EdgeType as E, VertexType as V


@dataclass(frozen=True)
class Action:
    kind: str
    vertices: tuple
    phase: Fraction = Fraction(0)
    subset: tuple = ()

    def record(self):
        return dict(kind=self.kind, vertices=list(self.vertices),
                    phase=str(self.phase), subset=list(self.subset))


def gadgets(graph):
    """Z phase gadgets, including Clifford leaves; only zero-phase roots."""
    result = []
    for leaf in graph.vertices():
        if graph.type(leaf) != V.Z or not graph.phase(leaf) or graph.vertex_degree(leaf) != 1:
            continue
        root = next(iter(graph.neighbors(leaf)))
        targets = tuple(sorted(set(graph.neighbors(root)) - {leaf}))
        if (graph.type(root) == V.Z and graph.phase(root) == 0 and targets
                and graph.edge_type(graph.edge(root, leaf)) == E.HADAMARD
                and all(graph.type(t) == V.Z and graph.edge_type(graph.edge(root, t)) == E.HADAMARD
                        for t in targets)):
            result.append((root, leaf, targets))
    return sorted(result)


def add_gadget(graph, targets, phase):
    if not targets or any(graph.type(v) != V.Z for v in targets):
        raise ValueError("Gadget targets must be existing Z spiders")
    row = sum(graph.row(v) for v in targets) / len(targets)
    root = graph.add_vertex(V.Z, -1, row, 0)
    leaf = graph.add_vertex(V.Z, -2, row, Fraction(phase) % 2)
    graph.add_edge(graph.edge(root, leaf), E.HADAMARD)
    graph.add_edges([graph.edge(root, v) for v in targets], E.HADAMARD)
    return root, leaf


def nest_pattern(targets, sign):
    n = len(targets)
    if not 2 <= n <= 4 or sign not in (-1, 1):
        raise ValueError("Controlled spider nests use 2..4 targets and sign +/-1")
    for k in range(1, n + 1):
        phase = Fraction(sign * (-1) ** (k - 1) * 2, 2 ** (n - 1)) % 2
        for support in combinations(sorted(targets), k):
            yield support, phase


def nest_matches(graph, targets, sign):
    available = {}
    targets = set(targets)
    for root, leaf, support in gadgets(graph):
        if root not in targets and leaf not in targets:
            available.setdefault(support, []).append((root, leaf))
    matched, missing = [], []
    for support, phase in nest_pattern(tuple(sorted(targets)), sign):
        pair = next((p for p in available.get(support, [])
                     if Fraction(graph.phase(p[1])) % 2 == phase), None)
        if pair is None:
            missing.append((support, (-phase) % 2))
        else:
            matched.append((pair[0], pair[1], support))
    return matched, missing


def growth(graph, action):
    if action.kind in {"identity_insert", "phase_split", "spider_split", "gadget_split"}:
        return 2
    if action.kind == "phase_transfer":
        return 0
    if action.kind == "spider_nest":
        matched, missing = nest_matches(graph, action.vertices, int(action.phase))
        return 2 * (len(missing) - len(matched))
    raise ValueError(action.kind)


def apply(graph, action):
    """Return a new graph; never mutate an archived state."""
    g = graph.clone()
    kind = action.kind
    if kind == "identity_insert":
        a, b = action.vertices
        edge = g.edge(a, b)
        if not g.connected(a, b) or g.edge_type(edge) != E.HADAMARD:
            raise ValueError("Identity insertion requires a Hadamard edge")
        g.remove_edge(edge)
        row = (g.row(a) + g.row(b)) / 2
        x = g.add_vertex(V.Z, -1, row, 0)
        y = g.add_vertex(V.Z, -1, row, 0)
        g.add_edges([g.edge(a, x), g.edge(x, y), g.edge(y, b)], E.HADAMARD)
    elif kind in {"phase_split", "spider_split"}:
        v, = action.vertices
        moved = set(action.subset)
        if g.type(v) != V.Z or not moved.issubset(set(g.neighbors(v))):
            raise ValueError("Invalid Z-spider split")
        if kind == "phase_split" and moved:
            raise ValueError("Phase-only split cannot move neighbours")
        row = g.row(v)
        new = g.add_vertex(V.Z, g.qubit(v), row + .25, action.phase)
        bridge = g.add_vertex(V.Z, -1, row + .125, 0)
        g.set_phase(v, (Fraction(g.phase(v)) - action.phase) % 2)
        for w in moved:
            edge = g.edge(v, w)
            et = g.edge_type(edge)
            g.remove_edge(edge)
            g.add_edge(g.edge(new, w), et)
        g.add_edges([g.edge(v, bridge), g.edge(bridge, new)], E.HADAMARD)
    elif kind == "gadget_split":
        root, leaf = action.vertices
        match = next((t for r, l, t in gadgets(g) if (r, l) == (root, leaf)), None)
        if match is None:
            raise ValueError("Invalid gadget")
        g.set_phase(leaf, (Fraction(g.phase(leaf)) - action.phase) % 2)
        add_gadget(g, match, action.phase)
        g.scalar.add_power(len(match) - 1)
    elif kind == "phase_transfer":
        leaf1, leaf2 = action.vertices
        supports = {l: t for r, l, t in gadgets(g)}
        if leaf1 == leaf2 or supports.get(leaf1) != supports.get(leaf2) or leaf1 not in supports:
            raise ValueError("Phase transfer requires distinct equal-support gadgets")
        g.set_phase(leaf1, (Fraction(g.phase(leaf1)) - action.phase) % 2)
        g.set_phase(leaf2, (Fraction(g.phase(leaf2)) + action.phase) % 2)
    elif kind == "spider_nest":
        matched, missing = nest_matches(g, action.vertices, int(action.phase))
        before = sum(len(t) - 1 for _, _, t in matched)
        after = sum(len(t) - 1 for t, _ in missing)
        g.remove_vertices([v for r, l, _ in matched for v in (r, l)])
        for targets, phase in missing:
            add_gadget(g, targets, phase)
        # A k-target gadget contributes 2^((1-k)/2); compensate exactly.
        g.scalar.add_power(after - before)
    else:
        raise ValueError(kind)
    return g.copy()


def candidates(graph, max_vertices, cap=32):
    """Deterministic bounded proposals, interleaved by rule family.

    These bounds and the pi/4 phase menu deliberately sacrifice completeness.
    """
    families = {k: [] for k in ["identity_insert", "phase_split", "spider_split",
                               "gadget_split", "phase_transfer", "spider_nest"]}
    phases = (Fraction(1, 4), Fraction(-1, 4), Fraction(1, 2))
    for edge in sorted(graph.edges()):
        if graph.edge_type(edge) == E.HADAMARD:
            families["identity_insert"].append(Action("identity_insert", tuple(graph.edge_st(edge))))
    for v in sorted(graph.vertices()):
        if graph.type(v) != V.Z:
            continue
        neighbours = tuple(sorted(graph.neighbors(v)))
        for phase in phases:
            families["phase_split"].append(Action("phase_split", (v,), phase))
        if len(neighbours) >= 3:
            # Bounded, explicit partitions; no claim to enumerate all 2^degree splits.
            for part in (neighbours[:1], neighbours[-1:], neighbours[:len(neighbours)//2]):
                for phase in (Fraction(0), Fraction(1, 4)):
                    families["spider_split"].append(Action("spider_split", (v,), phase, part))
    gs = gadgets(graph)
    for root, leaf, targets in gs:
        for phase in phases:
            families["gadget_split"].append(Action("gadget_split", (root, leaf), phase))
    for a, b in combinations(gs, 2):
        if a[2] == b[2]:
            for phase in phases:
                families["phase_transfer"].append(Action("phase_transfer", (a[1], b[1]), phase))
    contexts = {t for _, _, t in gs if 2 <= len(t) <= 4}
    for a, b in combinations(gs[:24], 2):
        union = tuple(sorted(set(a[2]) | set(b[2])))
        if 2 <= len(union) <= 4:
            contexts.add(union)
    nests = []
    for targets in sorted(contexts):
        for sign in (-1, 1):
            action = Action("spider_nest", targets, Fraction(sign))
            matched, missing = nest_matches(graph, targets, sign)
            if matched:  # No empty-context, arbitrary nest insertion in search.
                nests.append((len(missing) - len(matched), action))
    families["spider_nest"] = [a for _, a in sorted(nests, key=lambda x: (x[0], x[1].vertices, x[1].phase))]
    result = []
    seen = set()
    offset = 0
    while len(result) < cap:
        available = False
        for items in families.values():
            if offset < len(items):
                available = True
                action = items[offset]
                if action not in seen and graph.num_vertices() + growth(graph, action) <= max_vertices:
                    seen.add(action)
                    result.append(action)
                    if len(result) == cap:
                        break
        if not available:
            break
        offset += 1
    return result
