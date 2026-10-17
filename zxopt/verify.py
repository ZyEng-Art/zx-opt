"""Bounded variable elimination for Z-spider diagrams, independent of rewrites.

One Boolean variable per spider, normalized H-edge factors, identity factors for
simple edges, and unary phase factors implement the ZX tensor semantics directly.
The width cap prevents a large temporary tensor from exhausting host memory.
"""
import numpy as np
from pyzx.utils import VertexType as V, EdgeType as E


def graph_matrix(graph, max_width=20):
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

    while interior:
        scopes = {v: tuple(sorted(set().union(*(set(s) for s, _ in factors if v in s)))) for v in interior}
        v = min(interior, key=lambda v: (len(scopes[v]), v))
        scope = scopes[v]
        joined = [f for f in factors if v in f[0]]
        factors = [f for f in factors if v not in f[0]]
        value = multiply(joined, scope).sum(axis=scope.index(v))
        factors.append((tuple(w for w in scope if w != v), value))
        interior.remove(v)
    scope = tuple(sorted(boundary))
    value = multiply(factors, scope)
    if scope:
        value = value.transpose([scope.index(v) for v in order])
    return (value.reshape(2 ** len(graph.outputs()), 2 ** len(graph.inputs()))
            * graph.scalar.to_number())


def matrix_error(reference, candidate):
    if reference.shape != candidate.shape:
        raise ValueError("Matrix shape changed")
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
