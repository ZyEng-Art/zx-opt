"""Load the frozen v49 oracle and environment without editing their sources."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy"
for folder in (LEGACY, LEGACY / "quarl_style_rlzx_v10"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))


def environment(input_path=None):
    from circuit_env_v25 import source_classes
    import pyzx as zx
    input_path = Path(input_path or ROOT / "inputs/barenco_tof_3.qasm")
    circuit = zx.Circuit.from_qasm(input_path.read_text())
    Env, _ = source_classes(LEGACY / "upstream_source_v1", input_path)
    env = Env(circuit.qubits, len(circuit.gates))
    env.device = "cpu"
    env.gate_type = "gates"
    env.reset()
    return env


def oracle(capacity=256):
    from completion_feedback_v37 import FeedbackCompletion
    return FeedbackCompletion(capacity, "clean")
