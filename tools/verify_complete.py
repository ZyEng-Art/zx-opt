"""Reproduce exact JPV coverage checks and barenco-only graph proof receipts."""
import argparse
import contextlib
import datetime
from fractions import Fraction as F
import hashlib
from itertools import product
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zxopt.complete import NAMES, Diagram, RuleAction, axiom, actions, exact_tensor
from zxopt.complete.adapter import from_pyzx, to_pyzx
from zxopt.complete.axioms import SOURCE
from zxopt.legacy import environment
from zxopt.verify import graph_matrix, qiskit_matrix_zx_order


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--paper", type=Path)
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    git = ["git", "--git-dir=" + str(ROOT/".git"), "--work-tree=" + str(ROOT)]
    commit = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(git + ["status", "--porcelain", "--untracked-files=no"], text=True).strip()
    source_paths = (list((ROOT/"zxopt/complete").glob("*.py"))
                    + list((ROOT/"tests").glob("test_*.py")) + [Path(__file__).resolve()])
    sources = {str(f.relative_to(ROOT)): sha(f) for f in sorted(source_paths)}
    protocol = dict(started_at=datetime.datetime.now().astimezone().isoformat(),
                    git_commit=commit, tracked_dirty=bool(dirty), sources=sources,
                    python=sys.version, platform=platform.platform(),
                    input_sha256=sha(ROOT/"inputs/barenco_tof_3.qasm"),
                    source=SOURCE, theorem="Theorem 1.1; Figure 1 except IV/ZO, plus Figure 2",
                    paper_sha256=sha(a.paper) if a.paper else None,
                    resource="local CPU; no GPU or node44 job", phase_domain="pi/4 multiples only",
                    proof_boundary="JPV theorem plus explicit representation/context coverage argument; no Lean/Coq formalization; finite tests alone are not a completeness proof")
    dump(out/"protocol.json", protocol)

    def flatten(suite):
        for test in suite:
            if isinstance(test, unittest.TestSuite):
                yield from flatten(test)
            else:
                yield test.id()
    suite = unittest.defaultTestLoader.discover(str(ROOT/"tests"))
    test_ids = list(flatten(suite))
    with (out/"unit-tests.txt").open("w") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    units = dict(tests_run=result.testsRun, test_ids=test_ids, passed=result.wasSuccessful(),
                 failures=result.failures, errors=result.errors, skipped=result.skipped)
    dump(out/"unit-tests.json", units)
    if not result.wasSuccessful():
        raise AssertionError("Unit suite failed; see unit-tests.txt")
    print(json.dumps(dict(event="unit_tests", count=result.testsRun, status="passed")), flush=True)

    phases = tuple(F(k, 4) for k in range(8))
    checked = {name: 0 for name in NAMES}
    catalog = []
    for color in ("Z", "X"):
        for name in NAMES:
            rule = axiom(name, color=color)
            catalog.append(dict(instance=rule.record(), left=rule.left.record(), right=rule.right.record()))
        for alpha, beta, left, right, links in product(phases, phases, range(3), range(3), (1, 2, 5)):
            rule = axiom("S", color=color, alpha=alpha, beta=beta,
                         left_legs=left, right_legs=right, links=links)
            assert exact_tensor(rule.left) == exact_tensor(rule.right), rule.record()
            checked["S"] += 1
        for alpha, beta, gamma in product(phases, repeat=3):
            rule = axiom("C", color=color, alpha=alpha, beta=beta, gamma=gamma)
            assert exact_tensor(rule.left) == exact_tensor(rule.right), rule.record()
            checked["C"] += 1
        for alpha, arity in product(phases, range(9)):
            rule = axiom("H", color=color, alpha=alpha, arity=arity)
            assert exact_tensor(rule.left) == exact_tensor(rule.right), rule.record()
            checked["H"] += 1
        for name, alpha in product(("K", "SUP"), phases):
            rule = axiom(name, color=color, alpha=alpha)
            assert exact_tensor(rule.left) == exact_tensor(rule.right), rule.record()
            checked[name] += 1
        for name in ("I", "CP", "B", "EU", "E", "BW"):
            rule = axiom(name, color=color)
            assert exact_tensor(rule.left) == exact_tensor(rule.right), rule.record()
            checked[name] += 1
        print(json.dumps(dict(event="exact_axioms", color=color, cumulative=sum(checked.values()), status="passed")), flush=True)
    dump(out/"axiom-catalog.json", catalog)
    dump(out/"exact-axioms.json", dict(status="passed", instances=sum(checked.values()), by_axiom=checked,
                                      arithmetic="Exact dyadic Q[zeta8]/(zeta8^4+1), full tensor equality including global phase/amplitude",
                                      infinite_schema_argument="S: repeated Kronecker deltas fuse for every finite arity/positive link multiplicity. H: H^2=I and the tensor definition of X for every finite arity.",
                                      finite_arity_checks="S legs 0..2, links 1/2/5; H arity 0..8; these test ranges do not cap the implementation"))

    # Same original circuit only. Do not import any historical optimized witness.
    from qiskit import qasm2
    import numpy as np
    env = environment(ROOT/"inputs/barenco_tof_3.qasm")
    graph = from_pyzx(env.graph)
    reference = exact_tensor(graph)
    n = len(env.graph.inputs())
    def as_matrix(tensor):
        # Kernel port order is inputs, outputs; matrices list outputs first.
        return np.array([complex(v) for v in tensor]).reshape(2**n, 2**n).T
    source_matrix = qiskit_matrix_zx_order(qasm2.loads((ROOT/"inputs/barenco_tof_3.qasm").read_text()))
    root_error = float(np.max(np.abs(as_matrix(reference) - graph_matrix(env.graph))))
    from zxopt.verify import matrix_error
    input_error = matrix_error(source_matrix, as_matrix(reference))
    assert root_error < 2e-12 and input_error < 2e-12
    dump(out/"barenco-root.json", graph.record())
    chain = []
    specifications = [(axiom("E"), "backward"), (axiom("I"), "backward"),
                      (axiom("S", alpha=F(1, 4), beta=F(-1, 4)), "backward"),
                      (axiom("H", alpha=F(1, 4), arity=2), "backward")]
    max_error = root_error
    for step, (rule, direction) in enumerate(specifications, 1):
        action = next(actions(graph, rule, direction))
        child = action.apply(graph)
        assert exact_tensor(child) == reference
        pg = to_pyzx(child, n)
        error = float(np.max(np.abs(graph_matrix(pg) - as_matrix(reference))))
        assert error < 2e-12
        max_error = max(max_error, error)
        receipt = dict(step=step, action=action.record(), child_sha256=child.digest(),
                       vertices=len(child.nodes), wires=len(child.edges),
                       exported_vertices=pg.num_vertices(), exact_operator="passed", export_error=error)
        replayed = RuleAction.from_record(json.loads(json.dumps(action.record()))).apply(graph)
        assert replayed.record() == child.record()
        dump(out/("barenco-step-%d.json" % step), child.record())
        chain.append(receipt); graph = child
        print(json.dumps(dict(event="barenco_rewrite", step=step, axiom=rule.name, status="passed")), flush=True)
    dump(out/"barenco-proof.json", dict(status="passed", input="inputs/barenco_tof_3.qasm", qubits=n,
                                       chain=chain, root_graph_error=root_error, input_operator_error=input_error,
                                       max_export_error=max_error, validation="Exact equality to root at every step; independent exported full operators and replay SHA", extraction_required=False))
    assert sources == {str(f.relative_to(ROOT)): sha(f) for f in sorted(source_paths)}
    dump(out/"verification.json", dict(status="passed", git_commit=commit,
                                       tracked_dirty=bool(dirty), unit_tests=result.testsRun,
                                       exact_axiom_instances=sum(checked.values()),
                                       real_circuit_rewrites=len(chain), best_gate_claim=None,
                                       scope="Clifford+T axiom kernel; existing bounded RL runner is unchanged",
                                       elapsed_seconds=time.perf_counter()-start,
                                       artifacts={f.name: sha(f) for f in sorted(out.glob("*")) if f.is_file()}))
    print(json.dumps(dict(status="passed", output=str(out), exact_axioms=sum(checked.values()),
                          unit_tests=result.testsRun, real_circuit_rewrites=len(chain))), flush=True)


if __name__ == "__main__":
    main()
