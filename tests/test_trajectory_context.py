"""Trajectory termination state is visible, without changing the default context."""
from types import SimpleNamespace
import unittest

from zxopt.general_search import context_for


class TrajectoryContextTest(unittest.TestCase):
    def test_neutral_counter_is_visible_only_when_enabled(self):
        state = SimpleNamespace(last_counts=dict(gates=38), unscored_depth=2,
                                cost=None, diagram=SimpleNamespace(nodes={0: None}))
        start = SimpleNamespace(last_counts=dict(gates=40))
        best = dict(cost=dict(gates=37))
        args = SimpleNamespace(horizon=70, max_unscored=8, max_growth=64, neutral_stop_steps=0)
        expected = [-.05, -.075, .5, .25, 0., 1., 0., 1.]
        self.assertEqual(context_for(state, start, best, 35, args, 40, 1, 16), expected)
        args.neutral_stop_steps = 32
        self.assertEqual(context_for(state, start, best, 35, args, 40, 1, 16), expected[:-1]+[.5])
        self.assertEqual(context_for(state, start, best, 35, args, 40, 1, 0)[-1], 0.)
