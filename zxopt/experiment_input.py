"""Portable input snapshots and identities for general-ZX experiments."""
import hashlib
from pathlib import Path

from .legacy import ROOT


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recorded_input(folder, protocol):
    """Resolve and verify a run's immutable input, including old barenco logs."""
    if "input_snapshot" in protocol:
        name = Path(protocol["input_snapshot"])
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("Input snapshot must stay inside its run")
        path = Path(folder)/name
    else:
        path = ROOT/"inputs/barenco_tof_3.qasm"
    if sha256(path) != protocol["input_sha256"]:
        raise ValueError("Recorded input digest mismatch: "+str(path))
    return path
