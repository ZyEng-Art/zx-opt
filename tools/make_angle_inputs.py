"""Deterministic small numeric-angle circuits; no injected cancellation witness."""
from pathlib import Path


def generate():
    import math
    from qiskit import QuantumCircuit, qasm2
    out = Path(__file__).resolve().parents[1]/"inputs"
    # Textbook four-qubit Fourier transform with controlled phase decomposed
    # into RZ/CX and final bit-order swaps. Includes pi/8 and pi/16 phases.
    qc = QuantumCircuit(4)
    for target in reversed(range(4)):
        qc.h(target)
        for control in reversed(range(target)):
            theta = math.pi/(2**(target-control))
            qc.rz(theta/2, control); qc.rz(theta/2, target)
            qc.cx(control, target); qc.rz(-theta/2, target); qc.cx(control, target)
    for i in range(2):
        qc.cx(i, 3-i); qc.cx(3-i, i); qc.cx(i, 3-i)
    (out/"qft_4_numeric.qasm").write_text(qasm2.dumps(qc)+"\n")
    # Two-layer Ising ring ansatz, with fixed (gamma,beta)=(.37,.23),(.61,.41).
    # RX represented as H-RZ-H; ZZ represented as CX-RZ-CX.
    qc = QuantumCircuit(4)
    for q in range(4):
        qc.h(q)
    for gamma, beta in ((.37, .23), (.61, .41)):
        for u, v in ((0, 1), (1, 2), (2, 3), (3, 0)):
            qc.cx(u, v); qc.rz(2*gamma, v); qc.cx(u, v)
        for q in range(4):
            qc.h(q); qc.rz(2*beta, q); qc.h(q)
    (out/"ising_4_numeric.qasm").write_text(qasm2.dumps(qc)+"\n")


if __name__ == "__main__":
    generate()
