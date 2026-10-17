import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyzx as zx
from qiskit import QuantumCircuit, qasm2

from zxopt.patience import GatePatience
from zxopt.scalable_verify import ZXReference, InconclusiveVerification, reference_for


class PatienceVerifyTest(unittest.TestCase):
    def test_strict_total_gate_reset_and_downtime_resume(self):
        clock = GatePatience(10800, 40, now=100)
        self.assertEqual(clock.remaining(now=200), 10700)
        self.assertFalse(clock.observe(40, now=300))
        self.assertFalse(clock.observe(41, now=400))
        self.assertEqual(clock.last_improvement_unix, 100)
        self.assertTrue(clock.observe(39, now=500))
        self.assertEqual(clock.remaining(now=600), 10700)
        resumed = GatePatience(10800, 40, now=10000, record=clock.record())
        self.assertEqual(resumed.gates, 39)
        self.assertEqual(resumed.remaining(now=10000), 1300)
        self.assertEqual(resumed.remaining(now=11300), 0)
        with self.assertRaises(ValueError):
            GatePatience(100, 40, record=clock.record())

    def test_zero_patience_disables_timeout(self):
        self.assertEqual(GatePatience(0, 8, now=0).remaining(now=1000000), float("inf"))

    def test_scalable_identity_fixed_wires_and_scalar_amplitude(self):
        circuit = QuantumCircuit(12)
        circuit.h(0); circuit.cx(0, 11); circuit.rz(np.pi/4, 8)
        text = qasm2.dumps(circuit)
        with tempfile.TemporaryDirectory() as tmp:
            reference = ZXReference(text, Path(tmp)/"checks.jsonl")
            graph = zx.Circuit.from_qasm(text).to_graph()
            self.assertEqual(reference.check_graph(graph), 0)
            scaled = graph.copy(); scaled.scalar.add_power(2)
            with self.assertRaises(InconclusiveVerification):
                reference.check_graph(scaled)
            phase = graph.copy(); phase.scalar.add_phase(1/2)
            self.assertEqual(reference.check_graph(phase), 0)
            permuted = graph.copy(); permuted.set_outputs(tuple(reversed(permuted.outputs())))
            with self.assertRaises(InconclusiveVerification):
                reference.check_graph(permuted)
            bad = circuit.copy(); bad.x(3)
            with self.assertRaises(InconclusiveVerification):
                reference.check_qasm(qasm2.dumps(bad))
            self.assertTrue((Path(tmp)/"checks.jsonl").exists())

    def test_adaptive_small_independent_matrix_large_reduction(self):
        small = QuantumCircuit(3); small.h(0); small.cx(0, 2)
        self.assertIsInstance(reference_for(qasm2.dumps(small), "adaptive"), np.ndarray)
        large = QuantumCircuit(36); large.h(0); large.cx(0, 35)
        ref = reference_for(qasm2.dumps(large), "adaptive")
        self.assertIsInstance(ref, ZXReference)
        self.assertEqual(ref.check_qasm(qasm2.dumps(large)), 0)

    def test_bounded_menu_is_deterministic_and_resumeable(self):
        from zxopt.legacy import environment
        from zxopt.general_kernel import GeneralKernel, Views
        from zxopt.complete.adapter import from_pyzx
        from zxopt.verify import graph_matrix
        env = environment()
        raw = from_pyzx(env.graph); view = Views(5, graph_matrix(env.graph)).get(raw)
        left, right = (GeneralKernel(env, seed=29, total_candidate_cap=16) for _ in range(2))
        first = left.prepare(raw, view, view.graph.num_vertices()+64)
        self.assertEqual(len(first), 16)
        self.assertEqual([a.record() for a in first], [a.record() for a in right.prepare(raw, view, view.graph.num_vertices()+64)])
        saved_cap, saved_jpv = dict(left.cap_rounds), dict(left.jpv.rounds)
        second = left.prepare(raw, view, view.graph.num_vertices()+64)
        right.cap_rounds, right.jpv.rounds = saved_cap, saved_jpv
        self.assertEqual([a.record() for a in second], [a.record() for a in right.prepare(raw, view, view.graph.num_vertices()+64)])
