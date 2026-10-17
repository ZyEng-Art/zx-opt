"""Inspect/import/apply/replay the complete Clifford+T rule kernel.

Examples are in docs/clifford-t-completeness.txt. This command never equates
an extraction failure with an invalid ZX state, or calls finite RL complete.
"""
import argparse
import itertools
import json
from pathlib import Path

from . import NAMES, GENERAL_NAMES, Axiom, Diagram, RuleAction, actions, axiom
from .angles import restore as restore_phase
from .adapter import from_pyzx, to_pyzx


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Explicit new destination; no accidental replacement of previous evidence.
    with path.open("x") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("catalog")
    q.add_argument("--general-zx", action="store_true")
    q = sub.add_parser("import-qasm")
    q.add_argument("input", type=Path); q.add_argument("output", type=Path)
    q.add_argument("--allow-arbitrary-angles", action="store_true")
    q = sub.add_parser("apply")
    q.add_argument("input", type=Path); q.add_argument("output", type=Path)
    q.add_argument("--axiom", choices=GENERAL_NAMES, required=True)
    q.add_argument("--general-zx", action="store_true")
    q.add_argument("--parameters", default="{}", help='JSON, e.g. {"alpha":"1/4","links":2}')
    q.add_argument("--direction", choices=("forward", "backward"), default="forward")
    q.add_argument("--match-index", type=int, default=0)
    q = sub.add_parser("replay")
    q.add_argument("input", type=Path); q.add_argument("receipt", type=Path); q.add_argument("output", type=Path)
    q = sub.add_parser("export-pyzx")
    q.add_argument("input", type=Path); q.add_argument("output", type=Path)
    q.add_argument("--num-inputs", type=int, required=True)
    q.add_argument("--evaluate-expressions", action="store_true")
    a = p.parse_args()
    if a.command == "catalog":
        print(json.dumps(dict(scope="General JPV schemata with checked constraints" if a.general_zx else "Clifford+T, JPV Theorem 1.1",
                              axioms=GENERAL_NAMES if a.general_zx else NAMES,
                              source="https://arxiv.org/abs/1903.06035v3",
                              directions=["forward", "backward"],
                              limitations="No finite RL-search or gate-optimality guarantee"), indent=2))
        return
    if a.command == "import-qasm":
        import pyzx as zx
        g = zx.Circuit.from_qasm(a.input.read_text()).to_graph()
        write(a.output, from_pyzx(g, allow_arbitrary=a.allow_arbitrary_angles).record())
        return
    g = Diagram.from_record(json.loads(a.input.read_text()))
    if a.command == "apply":
        if a.match_index < 0:
            p.error("--match-index must be nonnegative")
        params = json.loads(a.parameters)
        for key in ("alpha", "beta", "gamma", "theta1", "theta2", "theta3"):
            if key in params:
                params[key] = restore_phase(params[key])
        rule = axiom(a.axiom, domain="general" if a.general_zx else "clifford_t", **params)
        action = next(itertools.islice(actions(g, rule, a.direction), a.match_index, None), None)
        if action is None:
            p.error("No such match for this instantiated axiom")
        child = action.apply(g)
        receipt = dict(action=action.record(), child_sha256=child.digest())
        # Guard both output names before writing either artifact.
        receipt_path = Path(str(a.output) + ".receipt.json")
        if a.output.exists() or receipt_path.exists():
            raise FileExistsError("Choose new diagram and receipt paths")
        write(a.output, child.record()); write(receipt_path, receipt)
    elif a.command == "replay":
        receipt = json.loads(a.receipt.read_text())
        child = RuleAction.from_record(receipt["action"]).apply(g)
        if child.digest() != receipt["child_sha256"]:
            raise ValueError("Rewrite receipt child SHA mismatch")
        write(a.output, child.record())
    else:
        pg = to_pyzx(g, a.num_inputs, evaluate_expressions=a.evaluate_expressions)
        # The output is PyZX's native JSON, not a claimed extracted circuit.
        with a.output.open("x") as f:
            f.write(pg.to_json())
    print(json.dumps(dict(command=a.command, output=str(a.output), status="passed")))


if __name__ == "__main__":
    main()
