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
                           ("anchor_prior", "nodes"), ("anchor_resource_mask", False)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.read(self.package(**{key: value}))

    def test_nan_weights_cannot_enter_search(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.read(self.package(state_dict={"weight": torch.tensor([float('nan')])}))

    def test_digest_is_independent_of_dictionary_order_but_sensitive_to_values(self):
        a = {"x": torch.tensor([1.]), "y": torch.tensor([2.])}
        self.assertEqual(state_dict_digest(a), state_dict_digest(dict(reversed(list(a.items())))))
        self.assertNotEqual(state_dict_digest(a), state_dict_digest({**a, "y": torch.tensor([3.])}))
