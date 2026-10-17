"""Exact bridges between the general axiom kernel and the pinned PyZX graph."""
from fractions import Fraction as F

from .diagram import Diagram, z_network
from .ring import C8


def from_pyzx(graph):
    from pyzx.utils import VertexType as V, EdgeType as E
    d = Diagram()
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
        for _ in range(abs(power)):
            a, b = d.node("Z"), d.node("X")
            for _ in range(1 if power > 0 else 3):
                d.edge(a, b)

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


def to_pyzx(diagram, num_inputs):
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
    mapping = {v: g.add_vertex(V.BOUNDARY if n.kind == "B" else V.Z,
                              -1, 1, n.phase) for v, n in vertices.items()}
    for a, b, parity in edges:
        # Two fresh Z identities also work for self loops, unlike a single
        # subdivision, which would create duplicate edges in a simple backend.
        u, v = g.add_vertex(V.Z), g.add_vertex(V.Z)
        g.add_edge(g.edge(mapping[a], u), E.HADAMARD if parity else E.SIMPLE)
        g.add_edge(g.edge(u, v), E.SIMPLE)
        g.add_edge(g.edge(v, mapping[b]), E.SIMPLE)
    g.set_inputs(tuple(mapping[v] for v in diagram.ports[:num_inputs]))
    g.set_outputs(tuple(mapping[v] for v in diagram.ports[num_inputs:]))
    # PyZX's tensorfy follows drawing rows. Grouping all fresh identities before
    # their endpoints can create a huge disconnected tensor. A greedy cut order
    # provides sensible rows without changing topology or boundary ordering.
    processed = set(g.inputs())
    pending = set(g.vertices()) - processed - set(g.outputs())
    for i, v in enumerate(g.inputs()):
        g.set_row(v, 0); g.set_qubit(v, i)
    step = 0
    while pending:
        v = min(pending, key=lambda w: (sum(1 if u not in processed else -1 for u in g.neighbors(w)), w))
        step += 1
        g.set_row(v, step)
        processed.add(v); pending.remove(v)
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
