"""Indexed export layout must retain every greedy row and vertex-ID tie."""
from fractions import Fraction as F
import random
import unittest

import pyzx as zx
from pyzx.utils import VertexType as V, EdgeType as E

from zxopt.complete import Diagram
from zxopt.complete.adapter import _greedy_cut_order, to_pyzx
from zxopt.verify import graph_matrix


def reference_order(graph):
    processed = set(graph.inputs())
    pending = set(graph.vertices()) - processed - set(graph.outputs())
    result = []
    while pending:
        v = min(pending, key=lambda w: (sum(1 if u not in processed else -1
                                          for u in graph.neighbors(w)), w))
        result.append(v); processed.add(v); pending.remove(v)
    return result


class ExportLayoutTest(unittest.TestCase):
    def test_random_topologies_and_boundaries_preserve_exact_greedy_order(self):
        rng = random.Random(619)
        for size in (0, 1, 2, 5, 16, 65, 129):
            for repetition in range(10):
                graph = zx.Graph()
                vertices = [graph.add_vertex(V.Z) for _ in range(size)]
                if size > 2:
                    for a in vertices:
                        for b in vertices:
                            if a < b and rng.random() < 3/size:
                                graph.add_edge(graph.edge(a, b), E.HADAMARD)
                count = min(size//3, 3)
                rng.shuffle(vertices)
                graph.set_inputs(tuple(vertices[:count])); graph.set_outputs(tuple(vertices[count:2*count]))
                expected = reference_order(graph)
                self.assertEqual(_greedy_cut_order(graph), expected, (size, repetition))
                self.assertEqual(_greedy_cut_order(graph), expected)

    def test_parallel_wires_self_loops_and_disconnected_scalars_keep_rows(self):
        raw = Diagram(allow_arbitrary=True)
        a = raw.node('Z', F(1,7)); b = raw.node('X', F(2,9))
        raw.edge(a,b); raw.edge(a,b); raw.edge(a,a)
        raw.leg(a); raw.leg(b); raw.node('Z', F(1,3)); raw.loops = 1
        graph = to_pyzx(raw, 1)
        self.assertEqual([v for v in graph.vertices() if v not in graph.inputs()+graph.outputs()],
                         sorted(reference_order(graph)))
        for row, vertex in enumerate(reference_order(graph), 1):
            self.assertEqual(graph.row(vertex), row)
        self.assertEqual(tuple(graph.qubit(v) for v in graph.inputs()), (0,))
        self.assertEqual(tuple(graph.qubit(v) for v in graph.outputs()), (0,))
        self.assertEqual(graph_matrix(graph).shape, (2,2))
