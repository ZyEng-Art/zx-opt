"""Mixed-angle inputs retain their phases while finite JPV matches stay exact."""
from fractions import Fraction as F
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from qiskit import qasm2

from zxopt.complete import NAMES, Diagram, Node, RuleAction, axiom, exact_tensor, tensor_product
from zxopt.complete.adapter import from_pyzx, to_pyzx
from zxopt.complete.ring import is_quarter
from zxopt.general_feasibility import Feasibility
from zxopt.general_kernel import GeneralKernel, Views
from zxopt.general_score import VerifiedCompletion
from zxopt.jpv_actions import Proposals, cancel_h_pairs, replay_h_pairs
from zxopt.legacy import environment
from zxopt.verify import graph_matrix, qiskit_matrix_zx_order


def angle_wire(phase):
    d = Diagram(allow_arbitrary=True)
    v = d.node("Z", phase)
    d.leg(v); d.leg(v)
    return d


class ArbitraryAngleTest(unittest.TestCase):
    def test_explicit_phase_mode_roundtrip_and_strict_exact_tensor(self):
        for phase in (F(1, 8), F(1, 3), F(9169213, 96019774)):
            with self.assertRaises(ValueError):
                Diagram().node("Z", phase)
            with self.assertRaises(ValueError):
                axiom("K", alpha=phase)
            d = angle_wire(phase)
            self.assertEqual(d.nodes[0].phase, phase)
            self.assertEqual(d.clone().record(), d.record())
            restored = Diagram.from_record(json.loads(json.dumps(d.record())))
            self.assertEqual(restored.digest(), d.digest())
            with self.assertRaises(ValueError):
                exact_tensor(d)
            record = d.record(); record.pop("allow_arbitrary")
            with self.assertRaises(ValueError):
                Diagram.from_record(record)
        # General representation does not change the exact CT evaluator.
        self.assertEqual(exact_tensor(angle_wire(F(1, 4))),
                         exact_tensor(axiom("H", alpha=F(1, 4), arity=2).right))
        self.assertNotIn("allow_arbitrary", axiom("I").left.record())

    def test_pyzx_roundtrip_does_not_quantize_angles_or_scalar_phases(self):
        import pyzx as zx
        from pyzx.utils import VertexType as V, EdgeType as E
        for phase in (F(1, 8), F(1, 3), F(9169213, 96019774)):
            g = zx.Graph()
            i, v, o = g.add_vertex(V.BOUNDARY), g.add_vertex(V.Z, phase=phase), g.add_vertex(V.BOUNDARY)
            g.add_edge(g.edge(i, v), E.SIMPLE); g.add_edge(g.edge(v, o), E.SIMPLE)
            g.set_inputs((i,)); g.set_outputs((o,))
            g.scalar.add_phase(F(1, 7))
            with self.assertRaises(ValueError):
                from_pyzx(g)
            d = from_pyzx(g, allow_arbitrary=True)
            self.assertIn(phase, [n.phase for n in d.nodes.values()])
            np.testing.assert_allclose(graph_matrix(to_pyzx(d, 1)), graph_matrix(g), atol=2e-12)

    def test_all_axiom_families_replay_inside_arbitrary_angle_context(self):
        proposals = Proposals(seed=29, cap=128, matching_work=16384)
        checked = 0
        for name in NAMES:
            for color in ("Z", "X"):
                rule = axiom(name, color=color)
                for direction in ("forward", "backward"):
                    source = rule.left if direction == "forward" else rule.right
                    host = tensor_product(source, angle_wire(F(1, 3)))
                    reference = graph_matrix(to_pyzx(host, len(host.ports)//2))
                    generated = proposals.build(host)
                    relevant = [a for a in generated if a.axiom.name == name and a.direction == direction]
                    self.assertTrue(relevant, (name, color, direction))
                    for action in relevant:
                        child = RuleAction.from_record(json.loads(json.dumps(action.record()))).apply(host)
                        self.assertTrue(child.allow_arbitrary)
                        self.assertEqual(child.nodes[max(host.nodes)-2].phase, F(1, 3))
                        child, receipts = cancel_h_pairs(child)
                        self.assertEqual(replay_h_pairs(action.apply(host), receipts).digest(), child.digest())
                        np.testing.assert_allclose(
                            graph_matrix(to_pyzx(child, len(child.ports)//2)), reference, atol=2e-11)
                        checked += 1
        self.assertGreaterEqual(checked, 44)

    def test_non_ct_matches_are_skipped_without_truncating_phase_signatures(self):
        host = axiom("K", alpha=F(0)).left
        host = Diagram.from_record(dict(host.record(), allow_arbitrary=True))
        # int((pi/8)/pi*4) would be 0 and could alias the zero-phase signature.
        v = next(v for v, n in host.nodes.items() if n.kind == "X")
        host.nodes[v] = Node("X", F(1, 8), True)
        proposals = Proposals(seed=11, cap=128, matching_work=16384)
        actions = proposals.build(host)
        self.assertTrue(actions)
        self.assertGreater(proposals.stats["non_clifford_matches_skipped"], 0)
        for action in actions:
            self.assertTrue(all(is_quarter(host.nodes[w].phase) for _, w in action.match.vertices))
            child = action.apply(host)
            self.assertEqual(child.nodes[v].phase, F(1, 8))

    def test_mixed_input_native_and_jpv_steps_previews_scoring_and_masks(self):
        text = (Path(__file__).parent/"fixtures/mixed_angles.qasm").read_text()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"mixed.qasm"; p.write_text(text)
            env = environment(p)
            raw = from_pyzx(env.graph, allow_arbitrary=True)
            reference = qiskit_matrix_zx_order(qasm2.loads(text))
            views = Views(2, reference); view = views.get(raw)
            result = VerifiedCompletion(reference).evaluate(view, seconds=20)
            self.assertIsNotNone(result.result)
            kernel = GeneralKernel(env)
            available = kernel.prepare(raw, view, view.graph.num_vertices()+64)
            self.assertTrue(any(a.mode == "jpv" for a in available))
            self.assertTrue(any(a.mode == "legacy" for a in available))
            mask = Feasibility(len(raw.nodes)+64, len(raw.edges)+256)
            eligible, _ = mask.prepare(raw, view, available, kernel)
            self.assertTrue(eligible)
            for action in eligible:
                child, receipts = kernel.step(raw, view, action)
                preview, preview_receipts = kernel.preview(raw, view, action)
                self.assertEqual(child.record(), preview.record(), action.family)
                self.assertEqual(receipts, preview_receipts)
                self.assertTrue(child.allow_arbitrary)
                self.assertLess(views.get(child).error, 2e-8)


if __name__ == "__main__":
    unittest.main()
