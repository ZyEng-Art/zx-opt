"""General-diagram PPO, inferred JPV instances, and legal unscored bridges."""
from fractions import Fraction as F
import json
from pathlib import Path
import time
import unittest

import numpy as np
import torch

from zxopt.complete import NAMES, Diagram, RuleAction, axiom, exact_tensor
from zxopt.complete.adapter import from_pyzx
from zxopt.jpv_actions import Proposals, hh_identity, cancel_h_pairs, replay_h_pairs, connected_ports
from zxopt.general_archive import SearchState, GeneralArchive, following_state
from zxopt.general_kernel import Views, GeneralKernel
from zxopt.general_policy import GeneralAgent, observation, batch
from zxopt.legacy import environment
from zxopt.verify import graph_matrix
from zxopt.general_search import candidate_digest
from zxopt.general_score import VerifiedCompletion
from monitored_ppo import ppo_update


class GeneralTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.env = environment()
        cls.raw = from_pyzx(cls.env.graph)
        cls.views = Views(5, graph_matrix(cls.env.graph))
        cls.view = cls.views.get(cls.raw)

    def test_compiled_proposals_expose_every_axiom_both_directions(self):
        proposals = Proposals(seed=29, cap=128, matching_work=16384)
        for name in NAMES:
            rule = axiom(name)
            for direction in ("forward", "backward"):
                source = rule.left if direction == "forward" else rule.right
                generated = proposals.build(source)
                relevant = [a for a in generated if a.axiom.name == name and a.direction == direction]
                self.assertTrue(relevant, (name, direction))
                for a in relevant:
                    self.assertEqual(exact_tensor(a.apply(source)), exact_tensor(source))

    def test_deterministic_proposals_and_all_root_witnesses(self):
        left, right = Proposals(seed=11), Proposals(seed=11)
        for _ in range(2):
            aa, bb = left.build(self.raw), right.build(self.raw)
            self.assertEqual([a.record() for a in aa], [a.record() for a in bb])
            self.assertLessEqual(len(aa), 32)
            for a in aa:
                child = RuleAction.from_record(json.loads(json.dumps(a.record()))).apply(self.raw)
                child, receipts = cancel_h_pairs(child)
                self.assertEqual(replay_h_pairs(a.apply(self.raw), receipts).digest(), child.digest())
                self.assertLess(self.views.get(child).error, 2e-8)

    def test_hh_derivation_and_scalar_feedback(self):
        source, target, proof, _ = hh_identity()
        g = source
        for row in proof["proof"]:
            g = RuleAction.from_record(row).apply(g)
        self.assertTrue(g.isomorphic(target))
        circle = Diagram(); a, b = circle.node("H"), circle.node("H")
        circle.edge(a, b); circle.edge(a, b)
        child, receipts = cancel_h_pairs(circle)
        self.assertEqual(child.loops, 1)
        self.assertEqual(exact_tensor(child), exact_tensor(circle))
        self.assertEqual(replay_h_pairs(circle, receipts).record(), child.record())

    def test_legal_unscored_bridge_retention_and_reward_anchor(self):
        counts = dict(gates=40, cx=18)
        root = SearchState(axiom("I").left, counts, counts)
        child = axiom("H", arity=2).right
        unknown, reward = following_state(root, child, None)
        self.assertIsNone(unknown.cost)
        self.assertEqual(unknown.last_counts, counts)
        self.assertEqual(reward, 0.)
        pool = GeneralArchive(root, scored_capacity=2, latent_capacity=2)
        self.assertTrue(pool.add(unknown, counts))
        self.assertEqual(pool.summary()["best"], [40, 18])
        class PickLatent:
            def random(self): return .9
            def integers(self, upper): return 0
        restart, source = pool.choose(PickLatent())
        self.assertEqual(source, "latent")
        self.assertIsNone(restart.cost)
        unknown2, _ = following_state(unknown, axiom("E").left, None)
        recovered, reward = following_state(unknown2, child, dict(cost=dict(gates=38, cx=16)))
        self.assertAlmostEqual(reward, .2)
        self.assertEqual(recovered.unscored_depth, 0)
        self.assertEqual(recovered.cost["gates"], 38)

    def test_raw_color_change_survives_independent_scoring_view(self):
        kernel = GeneralKernel(self.env)
        aa = kernel.prepare(self.raw, self.view, self.view.graph.num_vertices()+64)
        chosen = next(a for a in aa if a.family == "H:backward" and any(v in connected_ports(self.raw) for v in a.vertices))
        before = self.raw.digest()
        child, _ = kernel.step(self.raw, self.view, chosen)
        digest = child.digest()
        view = self.views.get(child)
        self.assertEqual(child.digest(), digest)
        self.assertEqual(self.raw.digest(), before)
        self.assertTrue(any(n.kind == "X" and v in connected_ports(child) for v, n in child.nodes.items()))
        self.assertIsNotNone(view.graph)

    def test_general_policy_density_gradients_and_ppo(self):
        kernel = GeneralKernel(self.env)
        aa = kernel.prepare(self.raw, self.view, self.view.graph.num_vertices()+64)
        obs = observation(self.raw, self.view.graph, aa, [0., 0., 0., 0., 1., 1., 0., 0.])
        other = observation(self.raw, self.view.graph, aa[:7], [.1, 0., .2, .1, 0., 1., 0., 1.])
        torch.manual_seed(11); net = GeneralAgent()
        packed = batch([obs, other]); selected = torch.tensor([len(aa)-1, 3])
        rng = torch.get_rng_state().clone()
        _, logs, entropy, values, _, _ = net.get_action_and_value(packed, action=selected)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        logits = net.actor(packed[0]).flatten()
        positions = (packed[0].y != -1).nonzero(as_tuple=True)[0]
        for i in range(2):
            scores = logits[positions[packed[0].batch[positions] == i]]
            torch.testing.assert_close(logs[i], scores.log_softmax(0)[selected[i]])
        (-logs.mean()).backward()
        self.assertGreater(float(net.actor_encoder.kind.weight.grad.abs().sum()), 0.)
        self.assertTrue(torch.isfinite(entropy).all())
        with torch.no_grad():
            _, lp, _, v, _, _ = net.get_action_and_value(batch([obs]), action=torch.tensor([0]))
        rollout = [(obs, 0, float(lp[0]), float(v[0]), .1 if i % 2 else -.1, i == 7) for i in range(8)]
        optimizer = torch.optim.Adam(net.parameters(), lr=.0003)
        update = ppo_update(net, optimizer, rollout, 0., dict(epochs=1, minibatch=4), time.perf_counter()+30)
        self.assertEqual(update["optimizer_steps"], 2)
        self.assertGreater(update["actor_parameter_l2_delta"], 0.)
        self.assertTrue(np.isfinite(update["mean_loss"]))
        self.assertEqual(candidate_digest(aa), candidate_digest(aa))

    def test_compact_exact_scalar_import_for_both_power_signs(self):
        import pyzx as zx
        for power in range(-12, 13):
            g = zx.Graph(); g.scalar.add_power(power)
            d = from_pyzx(g)
            self.assertLessEqual(len(d.nodes), max(2, power//2+2))
            np.testing.assert_allclose([complex(v) for v in exact_tensor(d)], [g.scalar.to_number()], atol=2e-12)

    def test_invalid_extraction_is_excluded_and_repaired_on_a_copy(self):
        fixture = Path(__file__).parent/"fixtures/barenco_gadget_split.json"
        d = Diagram.from_record(json.loads(fixture.read_text()))
        digest = d.digest(); view = self.views.get(d)
        completion = VerifiedCompletion(graph_matrix(self.env.graph))
        e = completion.evaluate(view, seconds=10)
        self.assertIsNotNone(e.result)
        self.assertEqual(e.branch, "clifford_repair")
        self.assertGreater(completion.summary()["invalid_extraction_attempts"], 0)
        self.assertLess(completion.summary()["max_valid_operator_error"], 2e-8)
        self.assertEqual(d.digest(), digest)
        again = completion.evaluate(view)
        self.assertTrue(again.hit)
        self.assertEqual(again.result["sha256"], e.result["sha256"])


if __name__ == "__main__":
    unittest.main()
