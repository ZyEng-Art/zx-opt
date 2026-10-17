import unittest
import torch
from zxopt.legacy import environment
from zxopt.kernel import Kernel
from zxopt.rules import candidates
from zxopt.policy import ExpandedAgent, observation, batch
from zxopt.dedup import StructuralIndex, action_key


class PolicyTest(unittest.TestCase):
    def test_new_actions_share_sampling_and_ppo_density(self):
        torch.set_num_threads(1)
        env = environment()
        kernel = Kernel(env)
        base, old = kernel.prepare(env.graph)
        extra = candidates(env.graph, env.graph.num_vertices() + 16)
        state, mapping, _ = StructuralIndex().intern(env.graph)
        state.attempted.add(action_key(old[0][2], mapping))
        obs, actions = observation(base, old, extra, [0.,0.,0.,1.], state.attempted, mapping)
        self.assertEqual(int((obs[0].y != -1).sum()), len(actions))
        self.assertEqual(int(obs[0].y[old[0][0]]), -1)
        self.assertTrue(bool((obs[0].action_kind > 0).any()))
        torch.manual_seed(11)
        net = ExpandedAgent()
        packed = batch([obs])
        chosen, log, _, _, _, _ = net.get_action_and_value(packed)
        _, replay_log, _, _, _, _ = net.get_action_and_value(packed, action=chosen)
        torch.testing.assert_close(log, replay_log)
        manual = net.actor(packed[0]).flatten()[packed[0].y != -1].log_softmax(0)[chosen[0]]
        torch.testing.assert_close(log[0], manual)
        (-log.mean()).backward()
        self.assertTrue(torch.isfinite(net.rule_embedding.weight.grad).all())
        self.assertGreater(float(net.rule_embedding.weight.grad.abs().sum()), 0.)
