"""Native CFlow arithmetic must preserve every score, type, order and rewrite."""
from itertools import combinations
import pickle
from pathlib import Path
import random
import shutil
import sysconfig
import tempfile
import unittest

import pyzx as zx
from pyzx.simplify import match_score_2Q_simp

from tools.build_native_scores import build
from zxopt.legacy import LEGACY
from zxopt.native_flow import enable_native_flow


class GraphStub:
    def __init__(self, ids): self.graph = {v: {} for v in ids}
    def add(self, a, b, kind=2): self.graph[a][b] = kind; self.graph[b][a] = kind
    def neighbors(self, v): return self.graph[v].keys()
    def connected(self, a, b): return b in self.graph[a]
    def edge(self, a, b): return (a, b)
    def edge_type(self, edge): return self.graph[edge[0]][edge[1]]


class NativeScoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which('c++') or not Path(sysconfig.get_path('include'), 'Python.h').exists():
            raise unittest.SkipTest('Optional native backend requires a compiler and Python headers')
        import cflow_exact
        cls.previous = cflow_exact.native_score_batch
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        metadata = build(Path(cls.directory.name)/'build')
        cls.library = Path(metadata['library'])
        enable_native_flow(cls.library)
        cls.score = staticmethod(cflow_exact.native_score_batch)
        cflow_exact.native_score_batch = cls.previous

    def test_random_scores_match_pinned_formulas_and_types(self):
        rng = random.Random(197)
        checked = 0
        for trial in range(40):
            ids = rng.sample(range(-200, 1000), rng.randrange(4, 18))
            graph = GraphStub(ids)
            for a, b in combinations(ids, 2):
                if rng.random() < .35: graph.add(a, b, rng.choice([1, 2]))
            matches = [(None, None, None)]
            def subsets(values):
                values = tuple(values)
                return [()] + list(combinations(values, 1))[:3] + list(combinations(values, 2))[:3]
            for v in ids:
                ns = tuple(graph.neighbors(v))
                matches += [((v, ns, u), None, None) for u in subsets(ns)]
            for a, b in combinations(ids, 2):
                if not graph.connected(a, b): continue
                us0 = subsets(v for v in graph.neighbors(a) if v != b)
                us1 = subsets(v for v in graph.neighbors(b) if v != a)
                matches += [(None, (a, b, (u0, u1)), None) for u0 in us0 for u1 in us1]
            matches += [(None, None, tuple(rng.sample(ids, 3))) for _ in range(20)]
            for weights in ((1, 1, 1), (2, 3, 4), (.5, -1.25, 2.0)):
                expected = [match_score_2Q_simp(graph, match, weights) for match in matches]
                actual = self.score(graph.graph, matches, weights)
                self.assertEqual(actual, expected)
                self.assertEqual([type(v) for v in actual], [type(v) for v in expected])
                checked += len(matches)
        self.assertGreater(checked, 10000)

    def test_scores_are_read_only_and_rebuilt_for_changed_graphs(self):
        graph = GraphStub([4, 91, 103, 700]); graph.add(4, 91); graph.add(4, 103); graph.add(4, 700)
        match = ((4, (91, 103, 700), ()), None, None)
        before = pickle.dumps(graph.graph)
        self.assertEqual(self.score(graph.graph, [match], (1, 1, 1)), [match_score_2Q_simp(graph, match)])
        self.assertEqual(pickle.dumps(graph.graph), before)
        for a, b in combinations((91, 103, 700), 2): graph.add(a, b)
        self.assertEqual(self.score(graph.graph, [match], (1, 1, 1)), [match_score_2Q_simp(graph, match)])

    def test_unsupported_inputs_raise_fallback_errors(self):
        with self.assertRaises(TypeError): self.score([], [], (1, 1, 1))
        with self.assertRaises(TypeError): self.score({0: {0: 2}}, [], (1, 1, 1))
        with self.assertRaises(TypeError): self.score({'v': {}}, [], (1, 1, 1))
        self.assertEqual(self.score({}, [], (1, 1, 1)), [])

    def test_actual_flow_preserves_full_graph_and_verifies_every_batch(self):
        import cflow_exact
        rng = random.Random(29)
        for trial in range(6):
            circuit = zx.Circuit(3)
            for _ in range(35):
                q = rng.randrange(3)
                kind = rng.choice(['HAD', 'T', 'CNOT', 'S'])
                if kind == 'CNOT':
                    circuit.add_gate(kind, q, (q+rng.choice([1, 2])) % 3)
                else: circuit.add_gate(kind, q)
            graph = circuit.to_graph(); zx.to_graph_like(graph)
            left, right = graph.clone(), graph.clone()
            cflow_exact.flow(left, native=False, verify=True, cFlow=True, max_lc_unfusions=2, max_p_unfusions=2)
            prior = cflow_exact.native_score_batch; counters = {}
            try:
                cflow_exact.native_score_batch = self.score
                cflow_exact.flow(right, verify=True, statistics=counters,
                                 cFlow=True, max_lc_unfusions=2, max_p_unfusions=2)
            finally: cflow_exact.native_score_batch = prior
            self.assertEqual(pickle.dumps(left, protocol=5), pickle.dumps(right, protocol=5))
            self.assertGreater(counters.get('native_batches', 0), 0)
            self.assertEqual(counters.get('native_fallback_batches', 0), 0)
            self.assertEqual(counters['native_scores'], counters['native_scores_verified'])
