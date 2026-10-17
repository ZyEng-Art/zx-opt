"""A faster audit retains full native predicates and the recorded survivor."""
from dataclasses import replace
from fractions import Fraction as F
import unittest

import pyzx as zx
from pyzx.utils import VertexType as V

from tools.audit_general import replay_native_rooted
from zxopt.complete.adapter import from_pyzx
from zxopt.general_kernel import GeneralKernel, Views
from zxopt.legacy import environment
from zxopt.rules import add_gadget
from zxopt.verify import graph_matrix


class RootedReplayTest(unittest.TestCase):
    def test_global_native_actions_have_identical_children_and_receipts(self):
        env = environment(); raw = from_pyzx(env.graph, allow_arbitrary=True)
        views = Views(5, graph_matrix(env.graph)); view = views.get(raw)
        kernel = GeneralKernel(env, jpv=False)
        actions = [a for a in kernel.prepare(raw, view, view.graph.num_vertices()+64) if a.mode == 'legacy']
        self.assertTrue(actions)
        before = raw.record(); kinds = set()
        for action in actions:
            expected, receipts = kernel.step(raw, view, action)
            actual, checked = replay_native_rooted(kernel, raw, view, action)
            self.assertEqual(actual.record(), expected.record(), action.family)
            self.assertEqual(checked, receipts)
            self.assertLess(views.get(actual).error, 2e-8)
            kinds.add(action.family)
        self.assertGreaterEqual(len(kinds), 3)
        self.assertEqual(raw.record(), before)

    def test_wrong_identifier_or_rule_label_is_rejected(self):
        env = environment(); raw = from_pyzx(env.graph, allow_arbitrary=True)
        view = Views(5, graph_matrix(env.graph)).get(raw); kernel = GeneralKernel(env, jpv=False)
        action = next(a for a in kernel.prepare(raw, view, view.graph.num_vertices()+64)
                      if a.mode == 'legacy' and a.family != 'GF')
        with self.assertRaises(ValueError):
            replay_native_rooted(kernel, raw, view, replace(action, identifier=123456789))
        with self.assertRaises(ValueError):
            replay_native_rooted(kernel, raw, view, replace(action, family='not-the-recorded-rule'))

    def test_separate_gadget_groups_preserve_recorded_surviving_root(self):
        graph = zx.Graph()
        ins = [graph.add_vertex(V.BOUNDARY) for _ in range(3)]
        outs = [graph.add_vertex(V.BOUNDARY) for _ in range(3)]
        graph.set_inputs(tuple(ins)); graph.set_outputs(tuple(outs))
        targets = [graph.add_vertex(V.Z) for _ in range(3)]
        for a, v, b in zip(ins, targets, outs):
            graph.add_edge(graph.edge(a, v)); graph.add_edge(graph.edge(v, b))
        for support in (targets[:2], targets[1:]):
            for phase in (F(1,4), F(3,4)): add_gadget(graph, support, phase)
        raw = from_pyzx(graph, allow_arbitrary=True)
        views = Views(3, graph_matrix(graph)); view = views.get(raw)
        kernel = GeneralKernel(environment(), jpv=False)
        actions = [a for a in kernel.prepare(raw, view, view.graph.num_vertices()+64) if a.family == 'GF']
        self.assertEqual(len(actions), 2)
        for action in actions:
            expected, receipts = kernel.step(raw, view, action)
            actual, checked = replay_native_rooted(kernel, raw, view, action)
            self.assertEqual(actual.record(), expected.record())
            self.assertEqual(checked, receipts)
            self.assertLess(views.get(actual).error, 2e-8)
