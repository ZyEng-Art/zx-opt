"""Circuit-group limits retain compact search diversity and correct costs."""
from dataclasses import dataclass
import unittest

from zxopt.general_archive import GeneralArchive, SearchState


@dataclass
class DiagramStub:
    name: str
    node_count: int
    edge_count: int

    def digest(self): return self.name
    @property
    def nodes(self): return range(self.node_count)
    @property
    def edges(self): return range(self.edge_count)


def state(name, nodes, gates=40, edges=None):
    cost = dict(gates=gates, cx=10)
    return SearchState(DiagramStub(name, nodes, nodes if edges is None else edges), cost, cost)


class ArchiveQasmGroupsTest(unittest.TestCase):
    def test_many_same_circuit_variants_do_not_displace_distinct_circuits(self):
        root = state("root", 30)
        pool = GeneralArchive(root, 6, 2, 2, "root-qasm")
        other = state("other", 20, 39)
        self.assertTrue(pool.add(other, root.cost, "other-qasm"))
        for i in range(20):
            pool.add(state("variant"+str(i), 40+i), root.cost, "root-qasm")
        self.assertIn(other.key, pool.scored)
        self.assertIn(root.key, pool.scored)
        self.assertEqual(pool.summary()["qasm_groups"], 2)
        self.assertLessEqual(sum(q == "root-qasm" for q in pool.signatures.values()), 2)
        self.assertEqual(pool.summary()["best"], [39, 10])
        self.assertEqual(len(pool.scored), 1+pool.insertions-pool.evictions)

    def test_smaller_diagram_replaces_larger_group_representative(self):
        root = state("root", 30)
        pool = GeneralArchive(root, 6, 0, 2, "q")
        self.assertTrue(pool.add(state("large", 60), root.cost, "q"))
        self.assertTrue(pool.add(state("small", 20), root.cost, "q"))
        self.assertEqual(set(pool.scored), {"root", "small"})
        self.assertFalse(pool.add(state("huge", 100), root.cost, "q"))
        self.assertEqual(set(pool.scored), {"root", "small"})

    def test_latent_states_need_no_completion_and_recover_into_a_group(self):
        root = state("root", 30)
        pool = GeneralArchive(root, 6, 2, 2, "q")
        value = state("latent", 40); value.cost = None
        self.assertTrue(pool.add(value, root.cost))
        self.assertIn(value.key, pool.latent)
        value.cost = dict(gates=39, cx=9)
        self.assertTrue(pool.add(value, root.cost, "new-qasm"))
        self.assertNotIn(value.key, pool.latent)
        self.assertEqual(pool.scored[value.key].cost, dict(gates=39, cx=9))

    def test_root_is_preserved_even_when_cap_is_one(self):
        root = state("root", 30)
        pool = GeneralArchive(root, 2, 0, 1, "q")
        self.assertFalse(pool.add(state("smaller", 10), root.cost, "q"))
        self.assertEqual(list(pool.scored), ["root"])

    def test_missing_completion_signature_is_rejected_without_pool_mutation(self):
        root = state("root", 30)
        pool = GeneralArchive(root, 4, 2, 2, "q")
        before = pool.summary()
        with self.assertRaises(ValueError):
            pool.add(state("new", 20, 39), root.cost)
        self.assertEqual(pool.summary(), before)
