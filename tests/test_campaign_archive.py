"""Terminal collection preserves failures and never relabels them as passes."""
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from tools.node_anchor_campaign import archive_records


class CampaignArchiveTest(unittest.TestCase):
    def test_failed_trial_and_ownership_are_checksum_sealed(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            trial = base/"source/runs/comparison/circuit/arm"
            trial.mkdir(parents=True)
            (trial/"failure.json").write_text('{"type": "RuntimeError"}\n')
            (base/"ownership.json").write_text('{"container_id":"owned"}\n')
            meta = archive_records(base, "abcdef1234567890", "failed")
            self.assertEqual(meta["study_status"], "failed")
            self.assertEqual(meta, json.loads((base/"archive.json").read_text()))
            archive = Path(meta["path"])
            self.assertEqual(meta["bytes"], archive.stat().st_size)
            self.assertEqual(meta["sha256"], hashlib.sha256(archive.read_bytes()).hexdigest())
            with tarfile.open(archive) as tar:
                self.assertEqual(json.load(tar.extractfile("runs/comparison/circuit/arm/failure.json")),
                                 {"type": "RuntimeError"})
                self.assertIn("ownership.json", tar.getnames())
            with self.assertRaises(FileExistsError):
                archive_records(base, "abcdef1234567890", "failed")

    def test_running_study_cannot_be_sealed_as_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                archive_records(Path(tmp), "abc", "running")
            self.assertEqual(list(Path(tmp).iterdir()), [])
