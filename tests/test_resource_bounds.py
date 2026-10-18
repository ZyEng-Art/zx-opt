"""Differential budgets, scalar bookkeeping and conservative HH handling."""
from fractions import Fraction as F
import unittest

from zxopt.complete import Diagram, axiom, actions
from zxopt.complete.adapter import from_pyzx
from zxopt.general_feasibility import Feasibility
from zxopt.general_kernel import GeneralKernel, HybridAction, View, Views
from zxopt.legacy import environment
from zxopt.rule_schedule import RuleSchedule
from zxopt.verify import graph_matrix


def hybrid(action):
    return HybridAction("jpv", action.axiom.name+":"+action.direction, action,
                        None, None, (), (F(0),)*3, (0.,)*3, 0, action.direction)


class ResourceBoundsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = environment()
        cls.raw = from_pyzx(cls.env.graph, allow_arbitrary=True)
        cls.view = Views(5, graph_matrix(cls.env.graph)).get(cls.raw)

    def test_all_axiom_directions_bound_the_real_child(self):
        kernel = GeneralKernel(self.env, general_zx=True)
        for name in ("S", "I", "CP", "B", "K", "EU", "H", "SUP", "E", "C", "BW", "A"):
            rule = axiom(name, domain="general")
            for direction in ("forward", "backward"):
                raw = rule.left if direction == "forward" else rule.right
                a = hybrid(next(actions(raw, rule, direction)))
                child, _ = kernel.preview(raw, View(None, None, 0.), a)
                bound = kernel.resource_upper_bound(raw, View(None, None, 0.), a, raw.digest())
                with self.subTest(name=name, direction=direction):
                    self.assertGreaterEqual(bound[0], len(child.nodes))
                    self.assertGreaterEqual(bound[1], len(child.edges))

    def test_macro_bounds_cover_scalar_power_changes(self):
        seen = set()
        for power in (-9, -2, -1, 0, 1, 2, 9):
            g = self.view.graph.copy(); g.scalar.power2 = power
            from pyzx.utils import VertexType as V
            from zxopt.rules import add_gadget
            targets = tuple(v for v in g.vertices() if g.type(v) == V.Z)[:2]
            for phase in (F(1, 4), F(1, 4), F(1)):
                add_gadget(g, targets, phase)
            view = View(g, "scalar-"+str(power), 0.)
            kernel = GeneralKernel(self.env, general_zx=True, candidate_cap=128)
            # The fake key belongs only to this isolated structural count test.
            from runtime import graph_key
            view.key = graph_key(g)
            for a in kernel.prepare(self.raw, view, 1000):
                if a.mode != "macro":
                    continue
                seen.add(a.family)
                child, _ = kernel.preview(self.raw, view, a)
                bound = kernel.resource_upper_bound(self.raw, view, a, self.raw.digest())
                self.assertGreaterEqual(bound[0], len(child.nodes), (power, a.family))
                self.assertGreaterEqual(bound[1], len(child.edges), (power, a.family))
        self.assertEqual(len(seen), 6)

    def test_same_eligible_menu_under_tight_and_loose_budgets(self):
        kernel = GeneralKernel(self.env, general_zx=True, total_candidate_cap=32)
        menu = kernel.prepare(self.raw, self.view, 200)
        for dn, de in ((0, 0), (1, 1), (4, 4), (64, 256)):
            nv, ne = len(self.raw.nodes)+dn, len(self.raw.edges)+de
            reference = Feasibility(nv, ne, capacity=0)
            fast = Feasibility(nv, ne, capacity=0, mode="bounds")
            left, _ = reference.prepare(self.raw, self.view, menu, kernel)
            right, _ = fast.prepare(self.raw, self.view, menu, kernel)
            self.assertEqual([a.record() for a in left], [a.record() for a in right])
        self.assertGreater(fast.stats["bound_accepts"], 0)
        self.assertLess(fast.stats["previews"], reference.stats["previews"])

    def test_an_oversized_upper_bound_is_not_a_rejection(self):
        raw = Diagram(); a, b = raw.node("H"), raw.node("H")
        raw.edge(a, b); raw.edge(a, b)
        action = hybrid(next(actions(raw, axiom("I"), "backward")))
        kernel = GeneralKernel(self.env, general_zx=True)
        mask = Feasibility(1, 1, mode="bounds")
        eligible, _ = mask.prepare(raw, View(None, None, 0.), [action], kernel)
        self.assertEqual(eligible, [action])
        self.assertEqual(mask.stats["bound_accepts"], 0)
        self.assertEqual(mask.stats["previews"], 1)

    def test_staged_schedule_resets_only_for_strict_gate_improvements(self):
        schedule = RuleSchedule("staged", 3, 2)
        phases = []
        for _ in range(10):
            phases.append(schedule.expanded()); schedule.observe(False)
        self.assertEqual(phases, [False, False, False, True, True]*2)
        restored = RuleSchedule("staged", 3, 2, schedule.record())
        self.assertEqual(restored.record(), schedule.record())
        restored.observe(True)
        self.assertEqual(restored.without_improvement, 0)

    def test_search_defaults_enable_bounds_and_keep_reference_available(self):
        from zxopt.general_search import parser
        default = parser().parse_args(["unused-output"])
        reference = parser().parse_args(["unused-output", "--feasibility-mode", "reference"])
        self.assertEqual(default.feasibility_mode, "bounds")
        self.assertEqual(reference.feasibility_mode, "reference")
        self.assertEqual(default.rule_schedule, "static")
        self.assertEqual(default.candidate_allocation, "balanced")

    def test_native_windows_skip_generation_and_quotas_preserve_diversity(self):
        from unittest.mock import patch
        kernel = GeneralKernel(self.env, general_zx=True, total_candidate_cap=32)
        with patch.object(kernel.jpv, "build", side_effect=AssertionError("unneeded JPV matching")), \
             patch("zxopt.general_kernel.candidates", side_effect=AssertionError("unneeded macro generation")):
            self.assertTrue(all(a.mode == "legacy" for a in kernel.prepare(self.raw, self.view, 200, expanded=False)))
        kernel = GeneralKernel(self.env, general_zx=True, total_candidate_cap=32,
                               candidate_allocation="native-heavy")
        menu = kernel.prepare(self.raw, self.view, 200)
        from collections import Counter
        counts = Counter(a.mode for a in menu)
        self.assertEqual(counts, dict(legacy=16, macro=8, jpv=8))
        self.assertGreater(len({a.family for a in menu if a.mode == "jpv"}), 2)


if __name__ == "__main__":
    unittest.main()
