"""Bounded variable elimination for Z-spider diagrams, independent of rewrites.

One Boolean variable per spider, normalized H-edge factors, identity factors for
simple edges, and unary phase factors implement the ZX tensor semantics directly.
The width cap prevents a large temporary tensor from exhausting host memory.
"""
import numpy as np
from pyzx.utils import VertexType as V, EdgeType as E


def graph_matrix(graph, max_width=20):
    if graph.scalar.is_unknown:
        raise ValueError("Cannot verify a graph with an unknown scalar")
    order = tuple(graph.outputs()) + tuple(graph.inputs())
    if len(order) != len(set(order)):
        raise ValueError("Boundary vertices must have distinct input/output roles")
    boundary = set(order)
    factors = []
    interior = []
    for v in graph.vertices():
        if graph.is_ground(v):
            raise ValueError("Ground/discard vertices are outside this pure-state verifier")
        if graph.type(v) == V.Z:
            interior.append(v)
            factors.append(((v,), np.array([1., np.exp(1j * np.pi * float(graph.phase(v)))])))
        elif graph.type(v) != V.BOUNDARY or v not in boundary:
            raise ValueError("Only Z spiders and ordered boundaries are supported")
    identity = np.eye(2, dtype=complex)
    hadamard = np.array([[1., 1.], [1., -1.]], dtype=complex) / np.sqrt(2.)
    for edge in graph.edges():
        a, b = sorted(graph.edge_st(edge))
        if a == b:
            raise ValueError("Self loops are outside this verifier")
        et = graph.edge_type(edge)
        if et not in (E.SIMPLE, E.HADAMARD):
            raise ValueError("Unsupported edge type")
        factors.append(((a, b), identity if et == E.SIMPLE else hadamard))

    def multiply(items, scope):
        if len(scope) > max_width:
            raise MemoryError("ZX verification width exceeds configured bound")
        result = np.ones((2,) * len(scope), dtype=complex)
        for variables, value in items:
            shape = tuple(2 if v in variables else 1 for v in scope)
            result *= value.reshape(shape)
        return result

    # The primal adjacency gives exactly the same current factor-union scope
    # as repeatedly scanning every factor for every remaining variable. Keep
    # insertion order and tie breaks identical to the reference implementation,
    # including the order of floating-point multiplications.
    active = dict(enumerate(factors))
    incident = {v: set() for v in graph.vertices()}
    adjacent = {v: set() for v in graph.vertices()}
    for i, (variables, _) in active.items():
        for v in variables:
            incident[v].add(i); adjacent[v].update(w for w in variables if w != v)
    next_id = len(active)
    while interior:
        v = min(interior, key=lambda v: (len(adjacent[v])+1, v))
        scope = tuple(sorted(adjacent[v] | {v}))
        joined = [active[i] for i in sorted(incident[v])]
        for i in sorted(incident[v]):
            variables, _ = active.pop(i)
            for w in variables:
                if w != v:
                    incident[w].remove(i)
        incident[v].clear()
        value = multiply(joined, scope).sum(axis=scope.index(v))
        remaining = tuple(w for w in scope if w != v)
        active[next_id] = (remaining, value)
        for w in remaining:
            incident[w].add(next_id)
            adjacent[w].discard(v)
            adjacent[w].update(u for u in remaining if u != w)
        next_id += 1
        interior.remove(v)
    scope = tuple(sorted(boundary))
    value = multiply(active.values(), scope)
    if scope:
        value = value.transpose([scope.index(v) for v in order])
    return (value.reshape(2 ** len(graph.outputs()), 2 ** len(graph.inputs()))
            * graph.scalar.to_number())


def matrix_error(reference, candidate):
    if reference.shape != candidate.shape:
        raise ValueError("Matrix shape changed")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Nonfinite operator entries")
    overlap = np.vdot(reference, candidate)
    phase = overlap / abs(overlap) if abs(overlap) else 1.
    # Align only a unit-modulus global phase; never normalize amplitudes.
    return float(np.max(np.abs(reference - candidate / phase)))


def qiskit_matrix_zx_order(circuit):
    """Convert basis-index convention only: named wires keep their identities.

    PyZX lists q0 as the most significant bit; Qiskit lists q0 as the least
    significant bit. Reverse both row and column bit strings, not wire labels.
    """
    from qiskit.quantum_info import Operator
    n = circuit.num_qubits
    order = [int(format(i, "0%db" % n)[::-1], 2) for i in range(2 ** n)]
    return Operator(circuit).data[np.ix_(order, order)]
