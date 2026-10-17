from fractions import Fraction as F
import unittest
import networkx as nx
import pyzx as zx
from pyzx.utils import VertexType as V, EdgeType as E
from zxopt.dedup import StructuralIndex, action_key
from zxopt.rules import Action


def graph(order=(0, 1, 2, 3)):
    g = zx.Graph()
    vs = {i: g.add_vertex(V.BOUNDARY if i in (0, 3) else V.Z, i, i, F(1, 4) if i == 1 else 0) for i in order}
    g.set_inputs((vs[0],)); g.set_outputs((vs[3],))
    for a, b in [(0, 1), (1, 2), (2, 3)]:
        g.add_edge(g.edge(vs[a], vs[b]), E.HADAMARD if a == 1 else E.SIMPLE)
    return g, vs


class DedupTest(unittest.TestCase):
    def test_relabel_layout_and_action_mapping(self):
        a, va = graph()
        b, vb = graph((3, 2, 0, 1))
        b.set_row(vb[1], 98)
        index = StructuralIndex()
        first, ma, _ = index.intern(a)
        second, mb, fresh = index.intern(b)
        self.assertIs(first, second); self.assertFalse(fresh)
        first.attempted.add(action_key(Action("LC", (va[1],)), ma))
        self.assertIn(action_key(Action("LC", (vb[1],)), mb), second.attempted)

    def test_phase_edge_scalar_and_wire_order_are_distinct(self):
        a, vs = graph()
        changes = []
        b = a.clone(); b.set_phase(vs[1], F(1, 2)); changes.append(b)
        b = a.clone(); b.set_edge_type(b.edge(vs[1], vs[2]), E.SIMPLE); changes.append(b)
        b = a.clone(); b.scalar.floatfactor = 1.000000001; changes.append(b)
        b = a.clone(); b.set_inputs(a.outputs()); b.set_outputs(a.inputs()); changes.append(b)
        index = StructuralIndex(bucket=lambda _: "deliberately_colliding")
        index.intern(a)
        for b in changes:
            self.assertTrue(index.intern(b)[2])

    def test_wl_collision_requires_exact_isomorphism(self):
        # Classic regular-graph WL collision: a six-cycle vs two disjoint triangles.
        index = StructuralIndex(bucket=lambda _: "collision")
        for edges in [list(nx.cycle_graph(6).edges()), [(0,1),(1,2),(2,0),(3,4),(4,5),(5,3)]]:
            g = zx.Graph()
            for _ in range(6): g.add_vertex(V.Z)
            g.add_edges(edges, E.HADAMARD)
            self.assertTrue(index.intern(g)[2])
        self.assertEqual(len(index.states), 2)


if __name__ == "__main__":
    unittest.main()
