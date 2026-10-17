"""Scalar-accurate matrix checks of identities, not held-out search circuits."""
from fractions import Fraction as F
from itertools import combinations
import unittest

import numpy as np
import pyzx as zx
from pyzx.utils import EdgeType as E, VertexType as V
from zxopt.rules import Action, apply, add_gadget, nest_pattern, candidates
from zxopt.verify import graph_matrix


def wires(n):
    g = zx.Graph()
    inputs, outputs, centres = [], [], []
    for q in range(n):
        i = g.add_vertex(V.BOUNDARY, q, 0)
        c = g.add_vertex(V.Z, q, 1, F(q, 4))
        o = g.add_vertex(V.BOUNDARY, q, 2)
        g.add_edges([g.edge(i, c), g.edge(c, o)], E.SIMPLE)
        inputs.append(i); outputs.append(o); centres.append(c)
    g.set_inputs(tuple(inputs)); g.set_outputs(tuple(outputs))
    return g, centres


class RulesTest(unittest.TestCase):
    def equivalent(self, before, after):
        np.testing.assert_allclose(graph_matrix(before), graph_matrix(after), atol=2e-12, rtol=2e-12)

    def test_independent_contraction_agrees_with_pyzx(self):
        for n in range(1, 4):
            g, nodes = wires(n)
            if n > 1:
                g.add_edge(g.edge(nodes[0], nodes[1]), E.HADAMARD)
            np.testing.assert_allclose(graph_matrix(g), g.to_matrix(), atol=1e-12)
        g, _ = wires(2)
        with self.assertRaises(MemoryError):
            graph_matrix(g, max_width=1)

    def test_all_split_phases_and_partitions(self):
        for phase in [F(0), F(1, 4), F(-1, 4), F(1, 2), F(3, 4)]:
            g, nodes = wires(3)
            g.add_edges([g.edge(nodes[0], nodes[1]), g.edge(nodes[0], nodes[2])], E.HADAMARD)
            original = g.to_json()
            for subset in [(), (nodes[1],), tuple(g.neighbors(nodes[0]))]:
                a = Action("phase_split" if not subset else "spider_split", (nodes[0],), phase, subset)
                self.equivalent(g, apply(g, a))
                self.assertEqual(g.to_json(), original)
            self.equivalent(g, apply(g, Action("identity_insert", (nodes[0], nodes[1]))))

    def test_gadget_split_scalars_for_every_arity(self):
        for n in range(1, 5):
            for phase in [F(1, 4), F(-1, 4), F(1, 2)]:
                g, nodes = wires(n)
                r, l = add_gadget(g, tuple(nodes), F(3, 4))
                self.equivalent(g, apply(g, Action("gadget_split", (r, l), phase)))

    def test_phase_transfer(self):
        g, nodes = wires(3)
        _, a = add_gadget(g, tuple(nodes), F(1, 4))
        _, b = add_gadget(g, tuple(nodes), F(3, 4))
        for phase in [F(1, 4), F(-1, 4), F(1, 2)]:
            self.equivalent(g, apply(g, Action("phase_transfer", (a, b), phase)))
        with self.assertRaises(ValueError):
            apply(g, Action("phase_transfer", (a, a), F(1, 4)))

    def test_full_and_partial_spider_nests(self):
        for n in range(2, 5):
            for sign in (-1, 1):
                for count in [0, 1, 2 ** n // 2, 2 ** n - 1]:
                    g, nodes = wires(n)
                    pattern = list(nest_pattern(tuple(nodes), sign))
                    for targets, phase in pattern[:count]:
                        add_gadget(g, targets, phase)
                    self.equivalent(g, apply(g, Action("spider_nest", tuple(nodes), F(sign))))

    def test_bounds(self):
        g, _ = wires(2)
        self.assertEqual(candidates(g, g.num_vertices(), cap=32), [])
        actions = candidates(g, g.num_vertices() + 2, cap=7)
        self.assertLessEqual(len(actions), 7)
        self.assertEqual(len(set(actions)), len(actions))
        for a in actions:
            self.assertLessEqual(apply(g, a).num_vertices(), g.num_vertices() + 2)
        g, nodes = wires(2)
        g.add_edge(g.edge(*nodes), E.HADAMARD)
        all_actions = candidates(g, g.num_vertices() + 2, cap=64)
        self.assertEqual(len(set(all_actions)), len(all_actions))


if __name__ == "__main__":
    unittest.main()
