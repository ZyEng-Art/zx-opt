"""An audit must follow the actual target after moving a run directory."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from zxopt.experiment_input import recorded_input


class InputTest(unittest.TestCase):
    def test_portable_snapshot_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); folder = root/"remote-run"; folder.mkdir()
            data = b'OPENQASM 2.0; include "qelib1.inc"; qreg q[7]; h q[6];'
            (folder/"input.qasm").write_bytes(data)
            protocol = dict(input_snapshot="input.qasm", input_sha256=hashlib.sha256(data).hexdigest())
            moved = root/"downloaded"; folder.rename(moved)
            self.assertEqual(recorded_input(moved, protocol).read_bytes(), data)
            (moved/"input.qasm").write_bytes(data.replace(b'q[6]', b'q[0]'))
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                recorded_input(moved, protocol)

    def test_snapshot_path_cannot_resolve_another_run(self):
        for name in ("../another/input.qasm", "/tmp/input.qasm"):
            with self.assertRaisesRegex(ValueError, "inside its run"):
                recorded_input("unused", dict(input_snapshot=name, input_sha256="unused"))
