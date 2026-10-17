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
