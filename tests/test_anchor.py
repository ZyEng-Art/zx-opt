"""Rooted coverage, proof replay and the sampled joint PPO distribution."""
from fractions import Fraction as F
import json
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Batch

from zxopt.anchor_policy import (Anchor, AnchorAgent, anchor_observation, batch,
                                 catalog, conditional_observation, empty_action)
from zxopt.anchor_ppo import ppo_update
from zxopt.complete import Diagram, axiom, exact_tensor
from zxopt.complete.diagram import tensor_product
from zxopt.complete.rewrite import matches
from zxopt.general_kernel import GeneralKernel, Views
from zxopt.general_feasibility import Feasibility
from zxopt.complete.adapter import from_pyzx
from zxopt.jpv_actions import Proposals
from zxopt.legacy import environment
from zxopt.verify import graph_matrix


class AnchorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.env = environment()
        cls.raw = from_pyzx(cls.env.graph, allow_arbitrary=True)
        cls.views = Views(5, graph_matrix(cls.env.graph))
        cls.view = cls.views.get(cls.raw)
        cls.context = [0., 0., 0., 0., 1., 1., 0., 0.]

    def test_rooted_embeddings_partition_the_global_matches(self):
        pattern = axiom("B").right
        host = tensor_product(pattern, pattern)
        root = min(v for v in pattern.nodes if v not in pattern.ports)
        global_rows = {json.dumps(m.record(), sort_keys=True) for m in matches(host, pattern)}
        anchored_rows = set()
        for v, n in host.nodes.items():
            if n.kind == "B":
                continue
            rows = list(matches(host, pattern, vertex_anchor=(root, v)))
            self.assertTrue(all(dict(m.vertices)[root] == v for m in rows))
            anchored_rows.update(json.dumps(m.record(), sort_keys=True) for m in rows)
        self.assertTrue(global_rows)
        self.assertEqual(global_rows, anchored_rows)

    def test_every_schema_direction_has_a_reachable_anchor(self):
        proposals = Proposals(cap=64, matching_work=32768, general=True)
        for name in ("S", "I", "CP", "B", "K", "EU", "H", "SUP", "E", "C", "BW", "A"):
            for direction in ("forward", "backward"):
                rule = axiom(name, domain="general")
                source = rule.left if direction == "forward" else rule.right
                internal = [v for v in source.nodes if v not in source.ports]
                if internal:
                    anchor = ("node", min(internal, key=lambda v: (-source.degree(v), source.nodes[v].kind, v)))
                else:
                    anchor = ("edge", 0) if source.edges else ("global", None)
                with self.subTest(name=name, direction=direction):
                    actions = [a for a in proposals.build(source, anchor)
                               if a.axiom.name == name and a.direction == direction]
                    self.assertTrue(actions)
                    for action in actions:
                        child = action.apply(source)
                        child.validate()
                        if len(source.nodes) < 8 and len(child.nodes) < 8:
                            self.assertEqual(exact_tensor(child), exact_tensor(source))

    def test_closed_wire_and_self_loop_insertion_proofs(self):
        g = Diagram(loops=1)
        actions = Proposals(general=True).build(g, ("loop", 0))
        identity = next(a for a in actions if a.axiom.name == "I")
        self.assertEqual(exact_tensor(identity.apply(g)), exact_tensor(g))
        g = Diagram(); v = g.node(); edge = g.edge(v, v)
        actions = Proposals(general=True).build(g, ("edge", edge))
        for action in actions:
            self.assertEqual(exact_tensor(action.apply(g)), exact_tensor(g))

    def test_general_angles_are_inferred_at_the_anchor(self):
        source = axiom("K", domain="general", alpha=F(1, 7)).left
        root = min((v for v in source.nodes if v not in source.ports),
                   key=lambda v: (-source.degree(v), source.nodes[v].kind, v))
        actions = Proposals(general=True).build(source, ("node", root))
        relevant = [a for a in actions if a.axiom.name == "K" and a.direction == "forward"]
        self.assertTrue(relevant)
        self.assertTrue(any(a.axiom.parameters["alpha"] == F(1, 7) for a in relevant))

    def test_catalog_and_anchor_encoding_do_not_query_matching(self):
        with patch.object(GeneralKernel, "prepare", side_effect=AssertionError("premature matching")):
            anchors = catalog(self.raw, self.view.graph)
            obs = anchor_observation(self.raw, self.view.graph, anchors, self.context)
        self.assertEqual(int((obs.y != -1).sum()), len(anchors))
        self.assertEqual({a.kind for a in anchors}, {"node", "edge", "global"})
        self.assertEqual({a.space for a in anchors}, {"raw", "view"})

    def test_local_kernel_does_not_use_whole_graph_native_prepare(self):
        kernel = GeneralKernel(self.env, general_zx=True, candidate_cap=8, total_candidate_cap=8)
        for anchor in (Anchor("raw", "node", 0), Anchor("raw", "edge", 0),
                       Anchor("raw", "global"),
                       next(a for a in catalog(self.raw, self.view.graph) if a.space == "view" and a.kind == "node")):
            with patch.object(kernel.native, "prepare", side_effect=AssertionError("whole graph query")):
                actions = kernel.prepare(self.raw, self.view, self.view.graph.num_vertices()+64, anchor.query())
            self.assertLessEqual(len(actions), 8)
            for action in actions:
                child, _ = kernel.preview(self.raw, self.view, action)
                self.assertLess(self.views.get(child).error, 2e-8)

    def test_native_rooted_union_and_preview_equal_global_rules(self):
        kernel = GeneralKernel(self.env, jpv=False)
        global_actions = kernel.prepare(self.raw, self.view, self.view.graph.num_vertices()+64)
        key = lambda a: (a.family, tuple(sorted(a.vertices)))
        expected = {key(a) for a in global_actions if a.mode == "legacy"}
        observed = set()
        for anchor in catalog(self.raw, self.view.graph, jpv=False):
            if anchor.kind != "node":
                continue
            actions = kernel.prepare(self.raw, self.view, self.view.graph.num_vertices()+64, anchor.query())
            for action in actions:
                if action.mode != "legacy":
                    continue
                observed.add(key(action))
                preview, _ = kernel.preview(self.raw, self.view, action)
                actual, _ = kernel.step(self.raw, self.view, action)
                self.assertEqual(preview.record(), actual.record(), action.family)
        self.assertEqual(expected, observed)

    def test_empty_menu_is_an_explicit_unchanged_state(self):
        kernel = GeneralKernel(self.env)
        child, receipts = kernel.step(self.raw, self.view, empty_action())
        self.assertEqual(child.record(), self.raw.record())
        self.assertEqual(receipts, [])
        self.assertIsNot(child, self.raw)

    def test_gadget_fusion_ids_are_remapped_from_the_local_menu(self):
        import pyzx as zx
        from pyzx.utils import VertexType as V
        from zxopt.rules import add_gadget
        from zxopt.kernel import Kernel
        graph = zx.Graph()
        inputs = [graph.add_vertex(V.BOUNDARY) for _ in range(3)]
        outputs = [graph.add_vertex(V.BOUNDARY) for _ in range(3)]
        graph.set_inputs(tuple(inputs)); graph.set_outputs(tuple(outputs))
        targets = [graph.add_vertex(V.Z, phase=0) for _ in range(3)]
        for a, v, b in zip(inputs, targets, outputs):
            graph.add_edge(graph.edge(a, v)); graph.add_edge(graph.edge(v, b))
        for support in (targets[:2], targets[1:]):
            for phase in (F(1, 4), F(3, 4)):
                add_gadget(graph, support, phase)
        native = Kernel(environment())
        _, global_actions = native.prepare(graph)
        global_groups = [a for _, _, a in global_actions if a.kind == "GF"]
        self.assertEqual(len(global_groups), 2)
        for group in global_groups:
            vertex = group.vertices[0]
            local = native.prepare_at(graph, vertex)
            selected = next((identifier, a) for _, identifier, a in local if a.kind == "GF")
            identifier, action = selected
            self.assertEqual(set(action.vertices), set(group.vertices))
            # The second global group also has local identifier -2.
            self.assertEqual(identifier, -2)
            actual = native.step(graph, action, identifier)
            self.assertLess(np.max(np.abs(graph_matrix(actual)-graph_matrix(graph))), 2e-8)

    def observations(self):
        anchors = catalog(self.raw, self.view.graph)
        kernel = GeneralKernel(self.env, general_zx=True, candidate_cap=8, total_candidate_cap=8)
        result = []
        for slot in (0, len(anchors)-1):
            anchor = anchors[slot]
            actions = kernel.prepare(self.raw, self.view, self.view.graph.num_vertices()+64, anchor.query())
            actions = actions or [empty_action()]
            a = anchor_observation(self.raw, self.view.graph, anchors, self.context)
            a.selected = torch.tensor([slot])
            result.append((a, *conditional_observation(self.raw, self.view.graph, actions, self.context, anchor)))
        return result

    def test_joint_density_batching_and_both_stage_gradients(self):
        observations = self.observations()
        torch.manual_seed(19); net = AnchorAgent(general_zx=True)
        packed = batch(observations); choices = torch.tensor([0, 0])
        state = torch.get_rng_state().clone()
        _, logs, entropy, values, la, lm = net.get_action_and_value(packed, action=choices)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        torch.testing.assert_close(logs, la+lm)
        for i in range(2):
            pa = packed[0]; pp = packed[1]
            anchor_logits = net.actor(pa).flatten()[(pa.y != -1) & (pa.batch == i)]
            local_logits = net.actor(pp).flatten()[(pp.y != -1) & (pp.batch == i)]
            torch.testing.assert_close(la[i], anchor_logits.log_softmax(0)[pa.selected[i]])
            torch.testing.assert_close(lm[i], local_logits.log_softmax(0)[choices[i]])
        (-la.mean()).backward(retain_graph=True)
        self.assertGreater(float(net.actor_head[-1].weight.grad.abs().sum()), 0.)
        net.zero_grad(); (-lm.mean()).backward()
        self.assertGreater(float(net.actor_head[-1].weight.grad.abs().sum()), 0.)
        self.assertTrue(torch.isfinite(entropy).all() and torch.isfinite(values).all())

    def test_ppo_changes_actor_with_joint_probability(self):
        observations = self.observations()
        torch.manual_seed(19); net = AnchorAgent(general_zx=True)
        with torch.no_grad():
            _, logs, _, values, _, _ = net.get_action_and_value(batch(observations), action=torch.tensor([0, 0]))
        rollout = [(observations[i % 2], 0, float(logs[i % 2]), float(values[i % 2]),
                    .2 if i % 2 else -.1, i == 7) for i in range(8)]
        update = ppo_update(net, torch.optim.Adam(net.parameters(), lr=.0003), rollout, 0.,
                            dict(epochs=2, minibatch=4), time.perf_counter()+30)
        self.assertEqual(update["optimizer_steps"], 4)
        self.assertGreater(update["actor_parameter_l2_delta"], 0.)
        self.assertTrue(np.isfinite(update["mean_loss"]))
        self.assertIn("anchor_entropy", update["metrics"])


if __name__ == "__main__":
    unittest.main()
