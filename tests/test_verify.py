import unittest
import numpy as np
import pyzx as zx
from qiskit import QuantumCircuit, qasm2
from zxopt.verify import graph_matrix, qiskit_matrix_zx_order, matrix_error


class VerifyTest(unittest.TestCase):
    def test_named_wire_conventions(self):
        c = QuantumCircuit(3)
        c.h(0); c.cx(0, 2); c.rz(.37, 1); c.x(2)
        g = zx.Circuit.from_qasm(qasm2.dumps(c)).to_graph()
        zx.to_graph_like(g)
        self.assertLess(matrix_error(qiskit_matrix_zx_order(c), graph_matrix(g)), 1e-12)
        wrong = QuantumCircuit(3)
        wrong.h(2); wrong.cx(2, 0); wrong.rz(.37, 1); wrong.x(0)
        self.assertGreater(matrix_error(qiskit_matrix_zx_order(wrong), graph_matrix(g)), .1)

    def test_amplitude_changes_are_not_global_phase(self):
        c = QuantumCircuit(1); c.h(0)
        m = qiskit_matrix_zx_order(c)
        self.assertGreater(matrix_error(m, 2*m), .5)
        self.assertLess(matrix_error(m, 1j*m), 1e-12)
        with self.assertRaises(ValueError):
            matrix_error(m, np.full_like(m, np.nan))

    def test_indexed_elimination_against_qiskit_random_named_wire_circuits(self):
        rng = np.random.default_rng(137)
        for n in range(1, 6):
            for repeat in range(5):
                circuit = QuantumCircuit(n)
                for _ in range(24):
                    wire = int(rng.integers(n)); gate = int(rng.integers(4))
                    if gate == 0:
                        circuit.h(wire)
                    elif gate == 1:
                        circuit.rz(float(rng.uniform(-np.pi, np.pi)), wire)
                    elif gate == 2:
                        circuit.x(wire)
                    elif n > 1:
                        other = (wire+int(rng.integers(1, n))) % n
                        circuit.cx(wire, other)
                graph = zx.Circuit.from_qasm(qasm2.dumps(circuit)).to_graph()
                zx.to_graph_like(graph)
                with self.subTest(n=n, repeat=repeat):
                    self.assertLess(matrix_error(qiskit_matrix_zx_order(circuit), graph_matrix(graph)), 2e-10)

    def test_indexed_elimination_retains_scalars_and_width_cap(self):
        from pyzx.utils import VertexType
        graph = zx.Graph()
        graph.add_vertex(VertexType.Z, phase=1/4)
        graph.add_vertex(VertexType.Z, phase=1/2)
        graph.scalar.add_power(-3)
        np.testing.assert_allclose(graph_matrix(graph), graph.to_matrix(), atol=1e-12)
        circuit = QuantumCircuit(2); circuit.h(0); circuit.cx(0, 1)
        graph = zx.Circuit.from_qasm(qasm2.dumps(circuit)).to_graph()
        zx.to_graph_like(graph)
        with self.assertRaises(MemoryError):
            graph_matrix(graph, max_width=2)
