"""Resumed weights remain input-bound, finite, and independently auditable."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import torch

from zxopt.model_snapshot import read_model_snapshot, state_dict_digest


class ModelSnapshotTest(unittest.TestCase):
    def package(self, **changes):
        value = dict(input_sha256="input", selection="anchor", anchor_prior="uniform",
                     anchor_resource_mask=True, state_dict={"weight": torch.tensor([1., 2.])}, steps=10)
        value.update(changes)
        return value

    def read(self, value):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"model.pt"; torch.save(value, path)
            return read_model_snapshot(path, "input", "anchor", "uniform", True)

    def test_hash_covers_the_exact_loaded_bytes_and_weights(self):
        saved, payload, info = self.read(self.package())
        self.assertEqual(hashlib.sha256(payload).hexdigest(), info["sha256"])
        self.assertEqual(state_dict_digest(saved["state_dict"]), info["tensor_sha256"])
        self.assertEqual(info["snapshot"], "resume-model.pt")

    def test_other_input_and_policy_support_are_rejected(self):
        for key, value in [("input_sha256", "other"), ("selection", "global"),
                           ("anchor_prior", "nodes"), ("anchor_resource_mask", False),
                           ("neutral_stop_steps", 32)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.read(self.package(**{key: value}))

    def test_rule_profiles_are_bound_and_old_capsules_default_to_full(self):
        self.read(self.package())
        with self.assertRaisesRegex(ValueError, "rule profile"):
            self.read(self.package(rule_profile="native"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"model.pt"; torch.save(self.package(rule_profile="native"), path)
            saved, _, _ = read_model_snapshot(path, "input", "anchor", "uniform", True, 0, "native")
            self.assertEqual(saved["rule_profile"], "native")

    def test_neutral_context_capsules_accept_only_the_same_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"model.pt"; torch.save(self.package(neutral_stop_steps=32), path)
            saved, _, _ = read_model_snapshot(path, "input", "anchor", "uniform", True, 32)
            self.assertEqual(saved["neutral_stop_steps"], 32)
            with self.assertRaisesRegex(ValueError, "neutral-stop context"):
                read_model_snapshot(path, "input", "anchor", "uniform", True, 64)

    def test_rule_allocation_is_bound_to_model_continuations(self):
        for key, value in (("rule_schedule", "staged"), ("native_steps", 32),
                           ("expanded_steps", 16), ("candidate_allocation", "native-heavy")):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "rule allocation"):
                self.read(self.package(**{key: value}))

    def test_nan_weights_cannot_enter_search(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.read(self.package(state_dict={"weight": torch.tensor([float('nan')])}))

    def test_digest_is_independent_of_dictionary_order_but_sensitive_to_values(self):
        a = {"x": torch.tensor([1.]), "y": torch.tensor([2.])}
        self.assertEqual(state_dict_digest(a), state_dict_digest(dict(reversed(list(a.items())))))
        self.assertNotEqual(state_dict_digest(a), state_dict_digest({**a, "y": torch.tensor([3.])}))
