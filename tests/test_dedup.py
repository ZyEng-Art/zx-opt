from fractions import Fraction as F
import unittest
import networkx as nx
import pyzx as zx
from pyzx.utils import VertexType as V, EdgeType as E
from zxopt.dedup import StructuralIndex, SearchIdentity, action_key
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

    def test_serialized_actions_do_not_prune_isomorphic_representations(self):
        from zxopt.legacy import ROOT  # Install frozen runtime search paths.
        from runtime import graph_key
        a, va = graph()
        b = a.clone()
        b.set_row(va[1], 98)
        self.assertEqual(graph_key(a), graph_key(a.clone()))
        self.assertNotEqual(graph_key(a), graph_key(b))
        index = StructuralIndex()
        state_a, ma, _ = index.intern(a)
        state_b, mb, fresh = index.intern(b)
        self.assertIs(state_a, state_b)
        self.assertFalse(fresh)
        identity = SearchIdentity("serialized")
        attempts_a, exact_ma = identity.actions(state_a, graph_key(a), ma)
        attempts_b, exact_mb = identity.actions(state_b, graph_key(b), mb)
        action = Action("LC", (va[1],))
        attempts_a.add(action_key(action, exact_ma))
        self.assertNotIn(action_key(action, exact_mb), attempts_b)
        again, _ = identity.actions(state_a, graph_key(a.clone()), ma)
        self.assertIs(attempts_a, again)
        self.assertNotEqual(identity.key(state_a, graph_key(a)), identity.key(state_b, graph_key(b)))
        structural = SearchIdentity("structural")
        self.assertEqual(structural.key(state_a, graph_key(a)), structural.key(state_b, graph_key(b)))

    def test_better_representative_replaces_payload_counts_and_root(self):
        from zxopt.archive import SearchBuffer
        from buffer import Entry
        def entry(gates, cx, payload):
            return Entry("same-isomorphism-class", dict(gates=gates, cx=cx), payload)
        root = entry(40, 20, "first")
        pool = SearchBuffer(root, replace_better=True)
        better = entry(36, 14, "useful-layout")
        self.assertTrue(pool.add(better, root.counts))
        self.assertIs(pool.root, better)
        self.assertIs(pool.entries[root.key], better)
        self.assertFalse(pool.add(entry(38, 12, "worse-gates"), root.counts))
        self.assertFalse(pool.add(entry(36, 14, "equal"), root.counts))
        self.assertTrue(pool.add(entry(36, 13, "better-tiebreak"), root.counts))
        self.assertEqual(pool.summary()["representative_replacements"], 2)
        self.assertEqual(len(pool.entries), 1)
        legacy = SearchBuffer(root)
        self.assertFalse(legacy.add(better, root.counts))
        self.assertIs(legacy.root, root)

    def test_representative_replacement_respects_trajectory_admission(self):
        from zxopt.archive import SearchBuffer
        from buffer import Entry
        root = Entry("root", dict(gates=40, cx=20), None)
        pool = SearchBuffer(root, capacity=3, replace_better=True)
        pool.add(Entry("other", dict(gates=39, cx=18), "old"), root.counts)
        self.assertFalse(pool.add(Entry("other", dict(gates=38, cx=17), "new"), dict(gates=37, cx=20)))
        self.assertEqual(pool.entries["other"].payload, "old")


if __name__ == "__main__":
    unittest.main()
