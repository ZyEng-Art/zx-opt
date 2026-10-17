"""Exact axioms, independent tensor semantics, and contextual rewrite witnesses."""
from fractions import Fraction as F
from itertools import islice, product
import json
import unittest

import numpy as np

from zxopt.complete import (NAMES, Axiom, Diagram, RuleAction, axiom, actions,
                            exact_tensor, tensor_product, join_ports, instances_at)
from zxopt.complete.adapter import from_pyzx, to_pyzx
from zxopt.complete.rewrite import Match, replace
from zxopt.complete.ring import C8, ONE, SQRT2, INV_SQRT2


def independent_tensor(d):
    """Direct incidence tensors; does not use z_network or exact-ring code."""
    args = []
    incidence = {v: [] for v in d.nodes}
    for i, edge in enumerate(d.edges):
        for v in edge:
            incidence[v].append(i)
    for v, n in d.nodes.items():
        labels = incidence[v]
        if n.kind == "B":
            args.extend([np.eye(2), labels + [len(d.edges) + d.ports.index(v)]])
            continue
        values = np.zeros((2,) * len(labels), dtype=complex)
        if n.kind == "H":
            values = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        else:
            phase = np.exp(1j * np.pi * float(n.phase))
            for bits in product((0, 1), repeat=len(labels)):
                if n.kind == "Z":
                    values[bits] = (int(all(b == 0 for b in bits))
                                    + phase * int(all(b == 1 for b in bits)))
                else:
                    values[bits] = (1 + phase * (-1) ** sum(bits)) * 2 ** (-len(labels)/2)
        args.extend([values, labels])
    if not args:
        return np.array([2 ** d.loops], dtype=complex)
    args.append([len(d.edges) + i for i in range(len(d.ports))])
    return np.asarray(np.einsum(*args, optimize=True)).reshape(-1) * 2 ** d.loops


class CompleteTest(unittest.TestCase):
    def test_exact_ring_and_phase_domain(self):
        self.assertEqual(SQRT2 * INV_SQRT2, ONE)
        self.assertEqual(SQRT2 * SQRT2, C8.of(2))
        self.assertEqual(C8.phase(1) * C8.phase(7), ONE)
        for phase in (F(1, 8), 0.25):
            with self.assertRaises(ValueError):
                axiom("K", alpha=phase)
        with self.assertRaises(ValueError):
            C8.of(F(1, 3))

    def test_all_axioms_both_colors_independent_semantics(self):
        for color, name in product(("Z", "X"), NAMES):
            a = axiom(name, color=color)
            l, r = exact_tensor(a.left), exact_tensor(a.right)
            self.assertEqual(l, r, (name, color))
            np.testing.assert_allclose([complex(v) for v in l], independent_tensor(a.left), atol=4e-14)
            np.testing.assert_allclose([complex(v) for v in r], independent_tensor(a.right), atol=4e-14)

    def test_all_clifford_t_phase_parameters(self):
        phases = [F(k, 4) for k in range(8)]
        for color in ("Z", "X"):
            for a, b, c in product(phases, repeat=3):
                rule = axiom("C", color=color, alpha=a, beta=b, gamma=c)
                self.assertEqual(exact_tensor(rule.left), exact_tensor(rule.right))
            for a in phases:
                for name in ("K", "SUP"):
                    rule = axiom(name, color=color, alpha=a)
                    self.assertEqual(exact_tensor(rule.left), exact_tensor(rule.right))

    def test_unbounded_arity_schema_representatives(self):
        for color in ("Z", "X"):
            for left, right, links in product(range(3), range(3), (1, 2, 5)):
                for a, b in product((F(0), F(1, 4), F(7, 4)), repeat=2):
                    rule = axiom("S", color=color, alpha=a, beta=b,
                                 left_legs=left, right_legs=right, links=links)
                    self.assertEqual(exact_tensor(rule.left), exact_tensor(rule.right))
            for arity in (0, 1, 2, 5, 8):
                rule = axiom("H", color=color, arity=arity)
                self.assertEqual(exact_tensor(rule.left), exact_tensor(rule.right))
        self.assertGreater(len(axiom("S", left_legs=40, right_legs=50, links=20).left.edges), 100)
        self.assertEqual(len(axiom("H", arity=70).left.ports), 70)

    def test_every_axiom_both_directions_and_serialized_receipts(self):
        for color, name, direction in product(("Z", "X"), NAMES, ("forward", "backward")):
            rule = axiom(name, color=color)
            source, target = (rule.left, rule.right) if direction == "forward" else (rule.right, rule.left)
            action = next(actions(source, rule, direction))
            row = json.loads(json.dumps(action.record()))
            child = RuleAction.from_record(row).apply(source)
            self.assertEqual(exact_tensor(source), exact_tensor(child), (color, name, direction))
            self.assertTrue(child.isomorphic(target), (color, name, direction))
            self.assertEqual(source.digest(), action.parent_sha256)

    def test_arbitrary_outside_contexts_and_feedback(self):
        for name in NAMES:
            rule = axiom(name)
            source = rule.left
            # Attach a nontrivial alternating-color context to every open port.
            context = Diagram()
            for k in range(len(source.ports)):
                v = context.node("X" if k % 2 else "Z", F(k, 4))
                context.leg(v); context.leg(v)
            host = tensor_product(source, context)
            n = len(source.ports)
            host = join_ports(host, [(host.ports[k], host.ports[n + 2*k]) for k in range(n)])
            action = next(actions(host, rule))
            child = action.apply(host)
            self.assertEqual(exact_tensor(host), exact_tensor(child), name)
        # Context directly joins two pattern ports, creating an internal edge.
        rule = axiom("S", left_legs=1, right_legs=1)
        traced = join_ports(rule.left, [(rule.left.ports[0], rule.left.ports[1])])
        child = next(actions(traced, rule)).apply(traced)
        self.assertEqual(exact_tensor(child), exact_tensor(traced))
        self.assertEqual(len(child.edges), 1)
        self.assertEqual(child.edges[0][0], child.edges[0][1])

    def test_identity_insertion_on_self_loop_and_bare_circle(self):
        rule = axiom("I")
        for host in (Diagram(loops=1), Diagram()):
            if not host.loops:
                v = host.node("Z", F(1, 4)); host.edge(v, v)
            child = next(actions(host, rule, "backward")).apply(host)
            self.assertEqual(exact_tensor(host), exact_tensor(child))
            restored = next(actions(child, rule)).apply(child)
            self.assertEqual(exact_tensor(host), exact_tensor(restored))
        h = Diagram(); v = h.node("H"); h.edge(v, v)
        self.assertFalse(exact_tensor(h)[0])  # tr(H) = 0
        h = Diagram(); a, b = h.node("H"), h.node("H"); h.edge(a, b); h.edge(a, b)
        self.assertEqual(exact_tensor(h), (C8.of(2),))  # tr(HH) = 2

    def test_empty_context_scalar_insertion_preserves_existing_graph(self):
        g = axiom("H", arity=3).right
        rule = axiom("E")
        a = next(actions(g, rule, "backward")); child = a.apply(g)
        self.assertEqual(len(child.nodes), len(g.nodes) + 2)
        self.assertEqual(exact_tensor(child), exact_tensor(g))

    def test_rejects_forged_stale_or_incomplete_matches(self):
        rule = axiom("S")
        g = rule.left
        a = next(actions(g, rule))
        changed = g.clone(); changed.node("Z")
        with self.assertRaises(ValueError):
            a.apply(changed)
        bad = Match(a.match.vertices, (), a.match.port_halves, a.match.wire_edges)
        with self.assertRaises(ValueError):
            replace(g, rule.left, rule.right, bad)
        with self.assertRaises(ValueError):
            list(actions(g, rule, "sideways"))
        corrupt = axiom("B").left.clone()
        # Delete the disconnected sqrt(2) component; exact equality must fail.
        corrupt.edges.pop(0); del corrupt.nodes[0]; del corrupt.nodes[1]
        self.assertNotEqual(exact_tensor(corrupt), exact_tensor(axiom("B").right))

    def test_pyzx_import_export_scalar_parallel_and_boundary_semantics(self):
        for name in NAMES:
            for d in (axiom(name).left, axiom(name).right):
                pg = to_pyzx(d, 0)
                numeric = np.asarray(pg.to_matrix()).reshape(-1)
                np.testing.assert_allclose(numeric, independent_tensor(d), atol=4e-14)
                self.assertEqual(exact_tensor(from_pyzx(pg)), exact_tensor(d))
        import pyzx as zx
        g = zx.Graph()
        g.scalar.add_power(-3); g.scalar.add_phase(F(3, 4)); g.scalar.add_node(F(1, 4))
        d = from_pyzx(g)
        np.testing.assert_allclose([complex(v) for v in exact_tensor(d)], [g.scalar.to_number()], atol=1e-14)

    def test_fair_stages_cover_phase_colors_and_large_arity(self):
        zero = list(instances_at(0))
        self.assertEqual({r.name for r in zero}, set(NAMES))
        self.assertEqual(len([r for r in zero if r.name == "C"]), 1024)
        self.assertEqual({r.parameters["color"] for r in zero}, {"Z", "X"})
        # Large instances are constructible; a finite test is not an arity bound.
        stage = islice(instances_at(5), 0, 1)
        self.assertEqual(next(stage).parameters["links"], 6)


if __name__ == "__main__":
    unittest.main()
