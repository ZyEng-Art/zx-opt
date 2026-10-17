"""Sufficient fixed-wire ZX identity reduction, with inconclusive failures.

This is an opt-in scalable checker, not a complete equivalence decision procedure
or an independently certified formal proof. Scalar modulus is checked numerically.
Small circuits retain the existing independent complete Qiskit matrix check.
"""
from collections import Counter, OrderedDict
import hashlib
import json
import math
from pathlib import Path
import time

import pyzx as zx
from pyzx.utils import EdgeType, VertexType
from qiskit import qasm2

from .verify import graph_matrix, matrix_error, qiskit_matrix_zx_order


class InconclusiveVerification(RuntimeError):
    pass


class ZXReference:
    def __init__(self, text, log=None, capacity=512):
        self.graph = zx.Circuit.from_qasm(text).to_graph()
        self.qubits = len(self.graph.inputs())
        self.input_sha256 = hashlib.sha256(text.encode()).hexdigest()
        self.log = Path(log) if log else None
        self.entries = OrderedDict(); self.capacity = capacity; self.stats = Counter()
        self.max_scalar_modulus_error = 0.
        if self.log:
            self.log.parent.mkdir(parents=True, exist_ok=True)

    def check_graph(self, graph, kind="graph", qasm_sha256=None):
        encoded = graph.to_json()
        key = hashlib.sha256(encoded.encode()).hexdigest()
        if key in self.entries:
            self.stats["cache_hits"] += 1; self.entries.move_to_end(key)
            passed, detail = self.entries[key]
            if not passed:
                raise InconclusiveVerification(detail)
            return 0.
        started = time.monotonic()
        miter = self.graph.adjoint(); miter.compose(graph)
        zx.full_reduce(miter, quiet=True)
        inputs, outputs = tuple(miter.inputs()), tuple(miter.outputs())
        wires = (len(inputs) == len(outputs) == self.qubits
                 and miter.num_vertices() == 2*self.qubits
                 and miter.num_edges() == self.qubits
                 and len(set(inputs+outputs)) == 2*self.qubits
                 and all(miter.type(v) == VertexType.BOUNDARY and miter.vertex_degree(v) == 1
                         for v in inputs+outputs)
                 and all(miter.connected(a, b) and miter.edge_type(miter.edge(a, b)) == EdgeType.SIMPLE
                         for a, b in zip(inputs, outputs)))
        scalar = miter.scalar
        value = scalar.to_number() if not scalar.is_unknown else complex(float("nan"))
        modulus_error = abs(abs(value)-1)
        passed = wires and not scalar.is_zero and math.isfinite(modulus_error) and modulus_error <= 1e-9
        receipt = dict(status="passed" if passed else "inconclusive", kind=kind,
                       input_sha256=self.input_sha256, candidate_graph_sha256=key,
                       qasm_sha256=qasm_sha256, qubits=self.qubits,
                       reduced_vertices=miter.num_vertices(), reduced_edges=miter.num_edges(),
                       named_identity_wires=wires, scalar_unknown=scalar.is_unknown,
                       scalar_zero=scalar.is_zero,
                       scalar_modulus_error=modulus_error if math.isfinite(modulus_error) else None,
                       seconds=time.monotonic()-started,
                       method="Pinned PyZX full_reduce(input.adjoint composed with candidate), ordered identity wires, scalar modulus <=1e-9; failure is inconclusive")
        if self.log:
            with self.log.open("a") as stream:
                stream.write(json.dumps(receipt, allow_nan=False)+"\n")
        self.stats[receipt["status"]] += 1
        detail = "ZX identity reduction inconclusive: "+json.dumps(receipt)
        self.entries[key] = (passed, detail)
        if len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
        if not passed:
            raise InconclusiveVerification(detail)
        self.max_scalar_modulus_error = max(self.max_scalar_modulus_error, modulus_error)
        return 0.  # certificate result, not a measured matrix error

    def check_qasm(self, text):
        return self.check_graph(zx.Circuit.from_qasm(text).to_graph(), "qasm", hashlib.sha256(text.encode()).hexdigest())


def reference_for(text, method="matrix", log=None):
    circuit = qasm2.loads(text)
    if method == "adaptive" and circuit.num_qubits > 7:
        return ZXReference(text, log)
    return qiskit_matrix_zx_order(circuit)


def check_graph(reference, graph):
    return (reference.check_graph(graph) if isinstance(reference, ZXReference)
            else matrix_error(reference, graph_matrix(graph)))


def check_qasm(reference, text):
    return (reference.check_qasm(text) if isinstance(reference, ZXReference)
            else matrix_error(reference, qiskit_matrix_zx_order(qasm2.loads(text))))
