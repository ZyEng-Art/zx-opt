"""Exact bridges between the general axiom kernel and the pinned PyZX graph."""
from fractions import Fraction as F
import heapq

from .diagram import Diagram, z_network
from .ring import C8
from .angles import Phase, numeric


def from_pyzx(graph, *, allow_arbitrary=False):
    from pyzx.utils import VertexType as V, EdgeType as E
    d = Diagram(allow_arbitrary=allow_arbitrary)
    kinds = {V.Z: "Z", V.X: "X", V.BOUNDARY: "B"}
    mapping = {}
    for v in graph.vertices():
        if graph.is_ground(v) or graph.type(v) not in kinds:
            raise ValueError("Only pure ZX spiders and boundaries are supported")
        mapping[v] = d.node(kinds[graph.type(v)], graph.phase(v))
    d.ports = tuple(mapping[v] for v in tuple(graph.inputs()) + tuple(graph.outputs()))
    for e in graph.edges():
        a, b = (mapping[v] for v in graph.edge_st(e))
        if graph.edge_type(e) == E.SIMPLE:
            d.edge(a, b)
        elif graph.edge_type(e) == E.HADAMARD:
            h = d.node("H"); d.edge(a, h); d.edge(h, b)
        else:
            raise ValueError("Unsupported wire type")
    scalar = graph.scalar
    if scalar.is_unknown:
        raise ValueError("Cannot import an unknown scalar")

    def root2(power):
        # Positive even powers use isolated Z_0 = 2. All negative powers fit in
        # one Z_0--X_0 parallel-wire component: m odd gives sqrt(2)^(2-m),
        # m even gives sqrt(2)^(4-m). This avoids counting 2*abs(power) nodes.
        if power < 0:
            a, b = d.node("Z"), d.node("X")
            for _ in range((2 if power % 2 else 4)-power):
                d.edge(a, b)
        else:
            for _ in range(power//2):
                d.node("Z")
            if power % 2:
                a, b = d.node("Z"), d.node("X"); d.edge(a, b)

    def phase(p):
        if not p:
            return
        # <Z_pi|X_alpha> = sqrt(2) exp(i*pi*alpha).
        a, b = d.node("Z", 1), d.node("X", p)
        d.edge(a, b); root2(-1)

    if scalar.is_zero or scalar.floatfactor == 0:
        d.node("Z", 1)
    else:
        root2(scalar.power2)
        phase(scalar.phase)
        for p in scalar.phasenodes:
            d.node("Z", p)
        factor = scalar.floatfactor
        if factor not in (1, -1, 1j, -1j):
            # Do not silently rationalize or discard a floating amplitude.
            raise ValueError("Unsupported floating scalar factor: import exact ZX scalars only")
        phase({1: F(0), -1: F(1), 1j: F(1, 2), -1j: F(3, 2)}[factor])
    return d.validate()


def _greedy_cut_order(graph):
    """The existing (cut change, vertex ID) order, with indexed integer scores."""
    processed = set(graph.inputs())
    pending = set(graph.vertices()) - processed - set(graph.outputs())
    scores = {v: sum(1 if u not in processed else -1 for u in graph.neighbors(v)) for v in pending}
    heap = [(score, v) for v, score in scores.items()]
    heapq.heapify(heap)
    order = []
    while pending:
        score, v = heapq.heappop(heap)
        if v not in pending or score != scores[v]:
            continue
        pending.remove(v); order.append(v)
        # Each newly processed neighbor changes exactly one +1 into -1.
        # Lazy stale entries retain the original integer and vertex-ID ties.
        for w in graph.neighbors(v):
            if w in pending:
                scores[w] -= 2
                heapq.heappush(heap, (scores[w], w))
    return order


def to_pyzx(diagram, num_inputs, *, evaluate_expressions=False):
    """Encode every general wire using fresh identities, preserving multiplicity.

    Export does not simplify, extract, permute named boundary wires, or refuse
    nonextractable intermediate diagrams. It returns a pure-Z PyZX graph.
    """
    import pyzx as zx
    from pyzx.utils import VertexType as V, EdgeType as E
    if not isinstance(num_inputs, int) or not 0 <= num_inputs <= len(diagram.ports):
        raise ValueError("Invalid input arity")
    vertices, edges, scalar = z_network(diagram)
    g = zx.Graph()
    def exported_phase(p):
        if isinstance(p, Phase):
            if not evaluate_expressions:
                raise ValueError("Symbolic phase export requires explicit evaluate_expressions=True")
            return numeric(p)
        return p
    mapping = {v: g.add_vertex(V.BOUNDARY if n.kind == "B" else V.Z,
                              -1, 1, exported_phase(n.phase)) for v, n in vertices.items()}
    existing = set()
    for a, b, parity in edges:
        pair = tuple(sorted((a, b)))
        if a != b and pair not in existing:
            g.add_edge(g.edge(mapping[a], mapping[b]), E.HADAMARD if parity else E.SIMPLE)
            existing.add(pair)
        else:
            # Only repeated wires/self loops need subdivisions in PyZX's simple
            # backend. Two identities preserve a self loop without parallel
            # collisions; unique wires keep their original compact form.
            u, v = g.add_vertex(V.Z), g.add_vertex(V.Z)
            g.add_edge(g.edge(mapping[a], u), E.HADAMARD if parity else E.SIMPLE)
            g.add_edge(g.edge(u, v), E.SIMPLE)
            g.add_edge(g.edge(v, mapping[b]), E.SIMPLE)
    g.set_inputs(tuple(mapping[v] for v in diagram.ports[:num_inputs]))
    g.set_outputs(tuple(mapping[v] for v in diagram.ports[num_inputs:]))
    # PyZX's tensorfy follows drawing rows. Grouping all fresh identities before
    # their endpoints can create a huge disconnected tensor. A greedy cut order
    # provides sensible rows without changing topology or boundary ordering.
    for i, v in enumerate(g.inputs()):
        g.set_row(v, 0); g.set_qubit(v, i)
    step = 0
    for step, v in enumerate(_greedy_cut_order(g), 1):
        g.set_row(v, step)
    for i, v in enumerate(g.outputs()):
        g.set_row(v, step+1); g.set_qubit(v, i)
    if not scalar:
        g.scalar.is_zero = True
    else:
        value = scalar.coefficients[0]
        if scalar != C8.of(value) or value.denominator != 1 or value < 1 or int(value) & (int(value)-1):
            raise AssertionError("H-chain normalization returned an unexpected scalar")
        g.scalar.add_power(2 * (int(value).bit_length()-1))
    return g
