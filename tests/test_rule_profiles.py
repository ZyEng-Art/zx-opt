"""Rule ablations remove generation work and retain default search semantics."""
import unittest
from unittest.mock import patch

from zxopt.complete import axiom
from zxopt.complete.adapter import from_pyzx
from zxopt.general_kernel import GeneralKernel, Views, rule_allowed
from zxopt.jpv_actions import Proposals
from zxopt.legacy import environment
from zxopt.verify import graph_matrix


class RuleProfilesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = environment()
        cls.raw = from_pyzx(cls.env.graph)
        cls.view = Views(5, graph_matrix(cls.env.graph)).get(cls.raw)

    def test_native_skips_macro_generation_and_jpv_initialization(self):
        with patch("zxopt.general_kernel.Proposals", side_effect=AssertionError("JPV constructed")), \
             patch("zxopt.general_kernel.candidates", side_effect=AssertionError("Macros constructed")):
            kernel = GeneralKernel(self.env, general_zx=True, rule_profile="native")
            actions = kernel.prepare(self.raw, self.view, 200)
            self.assertTrue(actions)
            self.assertTrue(all(a.mode == "legacy" for a in actions))

    def test_macros_retain_expansion_candidates_without_jpv(self):
        kernel = GeneralKernel(self.env, general_zx=True, rule_profile="macros")
        actions = kernel.prepare(self.raw, self.view, 200)
        self.assertIsNone(kernel.jpv)
        self.assertTrue(any(a.mode == "macro" for a in actions))
        self.assertTrue(all(a.mode != "jpv" for a in actions))

    def test_no_a_skips_the_matcher_before_construction(self):
        from zxopt.general_angles import compiled
        names = []
        def capture(name, *args):
            names.append(name)
            return compiled(name, *args)
        raw = axiom("A", domain="general").right
        with patch("zxopt.general_angles.compiled", side_effect=capture):
            Proposals(general=True, include_a=False, matching_work=16).build(raw)
        self.assertNotIn("A", names)
        names.clear()
        with patch("zxopt.general_angles.compiled", side_effect=capture):
            Proposals(general=True, matching_work=16).build(raw)
        self.assertIn("A", names)

    def test_optional_timers_do_not_change_proposals_or_rng(self):
        plain = Proposals(general=True, matching_work=64)
        timed = Proposals(general=True, matching_work=64, profile=True)
        for _ in range(2):
            self.assertEqual([a.record() for a in plain.build(self.raw)],
                             [a.record() for a in timed.build(self.raw)])
        self.assertNotIn("profile_seconds", plain.stats)
        self.assertGreater(timed.stats["profile_seconds"]["A:forward"], 0)

    def test_disabling_absent_a_keeps_other_candidates_exactly_paired(self):
        full = Proposals(general=True, matching_work=64)
        no_a = Proposals(general=True, include_a=False, matching_work=64)
        raw = axiom("I").left
        for _ in range(4):
            left, right = full.build(raw), no_a.build(raw)
            self.assertFalse(any(a.axiom.name == "A" for a in left))
            self.assertEqual([a.record() for a in left], [a.record() for a in right])

    def test_profile_guards_replay_and_invalid_profiles(self):
        self.assertTrue(rule_allowed("full", "jpv", "A:forward"))
        self.assertFalse(rule_allowed("no-a", "jpv", "A:backward"))
        self.assertFalse(rule_allowed("macros", "jpv", "S:forward"))
        with self.assertRaises(ValueError):
            GeneralKernel(self.env, rule_profile="unknown")
        macro = next(a for a in GeneralKernel(self.env, rule_profile="macros").prepare(
            self.raw, self.view, 200) if a.mode == "macro")
        with self.assertRaisesRegex(ValueError, "excluded"):
            GeneralKernel(self.env, rule_profile="native").step(self.raw, self.view, macro)

    def test_explicit_full_matches_default_actions(self):
        plain = GeneralKernel(self.env, general_zx=True, total_candidate_cap=32)
        full = GeneralKernel(self.env, general_zx=True, total_candidate_cap=32, rule_profile="full")
        for _ in range(2):
            self.assertEqual([a.record() for a in plain.prepare(self.raw, self.view, 200)],
                             [a.record() for a in full.prepare(self.raw, self.view, 200)])


if __name__ == "__main__":
    unittest.main()
