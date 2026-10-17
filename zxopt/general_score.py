"""Verified completion with isolated repairs for legal nonstandard ZX diagrams.

A wrong extracted circuit is quarantined, never assigned a gate reward or saved
as a candidate. The already-verified raw state can still continue. Rewrites and
normalization that change its operator remain fatal.
"""
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
import time

import pyzx as zx
from qiskit import qasm2

from .legacy import oracle
from .general_kernel import OperatorMismatch
from .search import operation_limit, atomic_json
from .verify import graph_matrix, matrix_error, qiskit_matrix_zx_order
from .scalable_verify import check_graph, check_qasm, InconclusiveVerification


class ExtractionMismatch(RuntimeError):
    pass


@dataclass
class Evaluation:
    result: object
    reason: object
    hit: bool
    branch: object
    failures: list


class VerifiedCompletion:
    def __init__(self, reference, quarantine=None, capacity=256):
        self.reference = reference; self.capacity = capacity
        self.primary = oracle(); self.repair = oracle()
        self.quarantine = Path(quarantine) if quarantine is not None else None
        if self.quarantine is not None:
            self.quarantine.mkdir(parents=True, exist_ok=True)
        self.entries = OrderedDict(); self.valid_qasm = {}; self.invalid_qasm = {}
        self.stats = Counter(); self.max_valid_error = self.max_invalid_error = 0.

    def check(self, result, view_key, branch):
        sha = result["sha256"]
        if sha in self.valid_qasm:
            return
        error = self.invalid_qasm.get(sha)
        if error is None:
            try:
                error = check_qasm(self.reference, result["qasm"])
            except InconclusiveVerification as exc:
                self.stats["inconclusive_extractions"] += 1
                if self.quarantine is not None:
                    pending = self.quarantine.parent/"unverified-extractions"
                    pending.mkdir(exist_ok=True)
                    (pending/(sha+".qasm")).write_text(result["qasm"])
                    atomic_json(pending/(sha+".json"), dict(status="inconclusive", reason=str(exc), cost_excluded=result["cost"]))
                raise
        if error > 2e-8:
            self.invalid_qasm[sha] = error
            self.max_invalid_error = max(self.max_invalid_error, error)
            self.stats["invalid_extraction_attempts"] += 1
            if self.quarantine is not None:
                (self.quarantine/(sha+".qasm")).write_text(result["qasm"])
                atomic_json(self.quarantine/(sha+".json"), dict(status="invalid_extraction", sha256=sha,
                            view_key=view_key, branch=branch, operator_error=error, cost_excluded=result["cost"]))
            raise ExtractionMismatch("Extracted operator mismatch (discarded): "+str(error))
        self.max_valid_error = max(self.max_valid_error, error)
        self.valid_qasm[sha] = error

    def evaluate(self, view, seconds=2.):
        if view.graph is None:
            return Evaluation(None, view.normalization_error, False, None, [])
        if view.key in self.entries:
            self.stats["cache_hits"] += 1; self.entries.move_to_end(view.key)
            e = self.entries[view.key]
            return Evaluation(e.result, e.reason, True, e.branch, e.failures)
        self.stats["requests"] += 1
        deadline = time.perf_counter()+seconds; failures = []; result = branch = None
        for branch in ("primary", "clifford_repair", "full_reduce_repair"):
            remaining = deadline-time.perf_counter()
            if remaining <= 0:
                failures.append(dict(branch=branch, reason="extraction time budget exhausted")); break
            try:
                with operation_limit(remaining):
                    graph = view.graph
                    scorer = self.primary
                    if branch != "primary":
                        graph = graph.clone()
                        if branch == "clifford_repair":
                            zx.simplify.clifford_simp(graph, quiet=True)
                        else:
                            zx.full_reduce(graph, quiet=True)
                        error = check_graph(self.reference, graph)
                        if error > 2e-8:
                            raise OperatorMismatch("Extraction repair changed the raw operator: "+str(error))
                        scorer = self.repair
                    _, candidate, _ = scorer.evaluate(graph)
                    self.check(candidate, view.key, branch)
                    result = candidate
                    self.stats[branch+"_successes"] += 1
                    break
            except OperatorMismatch:
                raise
            except Exception as exc:
                failures.append(dict(branch=branch, reason=type(exc).__name__+": "+str(exc)))
                self.stats[branch+"_failures"] += 1
                if isinstance(exc, TimeoutError):
                    break
        reason = None if result is not None else "; ".join(f["branch"]+": "+f["reason"] for f in failures)
        evaluation = Evaluation(result, reason, False, branch if result is not None else None, failures)
        self.entries[view.key] = evaluation
        if len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
        if result is None:
            self.stats["unscorable"] += 1
        return evaluation

    def summary(self):
        return dict(self.stats, unique_valid_qasm=len(self.valid_qasm), unique_invalid_qasm=len(self.invalid_qasm),
                    max_valid_operator_error=self.max_valid_error, max_rejected_operator_error=self.max_invalid_error)
