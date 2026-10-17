"""An old green audit cannot authenticate subsequently changed result files."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from tools.summarize_anchor import require_audit_receipt


class AuditReceiptTest(unittest.TestCase):
    def fixture(self, path):
        (path/"report.json").write_text('{"best":37}\n')
        (path/"events.jsonl").write_text('{"reward":0.1}\n')
        spec = dict(arm="anchor8_online", seed=11)
        receipt = dict(**spec, status="passed",
                       report_sha256=hashlib.sha256((path/"report.json").read_bytes()).hexdigest(),
                       events_sha256=hashlib.sha256((path/"events.jsonl").read_bytes()).hexdigest())
        return spec, dict(records=[receipt])

    def test_changed_report_or_events_invalidate_the_receipt(self):
        for filename in ("report.json", "events.jsonl"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp); spec, verification = self.fixture(path)
                self.assertEqual(require_audit_receipt(path, spec, verification)["status"], "passed")
                with (path/filename).open("a") as stream:
                    stream.write(" ")
                with self.assertRaises(ValueError):
                    require_audit_receipt(path, spec, verification)

    def test_another_seed_cannot_reuse_a_matching_file_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); spec, verification = self.fixture(path)
            spec["seed"] = 29
            with self.assertRaises(ValueError):
                require_audit_receipt(path, spec, verification)

    def test_changed_policy_snapshot_invalidates_the_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); spec, verification = self.fixture(path)
            model = path/"improvement-models/g37-t100.pt"
            model.parent.mkdir(); model.write_bytes(b"original audited model bytes")
            verification["records"][0]["improvement_model_receipts"] = {
                "improvement-models/g37-t100.pt": hashlib.sha256(model.read_bytes()).hexdigest()}
            require_audit_receipt(path, spec, verification)
            model.write_bytes(b"different model")
            with self.assertRaises(ValueError):
                require_audit_receipt(path, spec, verification)

    def test_policy_snapshot_cannot_escape_the_trial_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); spec, verification = self.fixture(path)
            verification["records"][0]["improvement_model_receipts"] = {"../model.pt": "sha"}
            with self.assertRaises(ValueError):
                require_audit_receipt(path, spec, verification)
