"""PPO terminals describe actual stop, horizon, resource and latent boundaries."""
import unittest

from tools.audit_general import immediate_terminal


class EpisodeTerminalTest(unittest.TestCase):
    def test_empty_anchor_stop_is_opt_in_and_does_not_change_normal_steps(self):
        config = dict(horizon=70, max_unscored=8)
        event = dict(status='anchor_noop', trajectory_step=1)
        self.assertFalse(immediate_terminal(event, config, 40))
        self.assertTrue(immediate_terminal(event, dict(config, anchor_noop_stop=True), 40))
        event.update(status='accepted', cost=dict(gates=39))
        self.assertFalse(immediate_terminal(event, dict(config, anchor_noop_stop=True), 40))

    def test_horizon_gate_excursion_and_latent_limits_are_distinct(self):
        config = dict(horizon=70, max_unscored=8)
        self.assertTrue(immediate_terminal(dict(status='rejected', trajectory_step=70), config, 40))
        for gates, expected in ((60, False), (61, True)):
            self.assertEqual(immediate_terminal(dict(status='accepted', trajectory_step=5,
                                                     cost=dict(gates=gates)), config, 40), expected)
        for depth, expected in ((7, False), (8, True)):
            self.assertEqual(immediate_terminal(dict(status='accepted_unscored', trajectory_step=5,
                                                     unscored_depth=depth), config, 40), expected)

    def test_neutral_boundary_is_separate_from_horizon_and_gate_cost(self):
        config = dict(horizon=70, max_unscored=8, neutral_stop_steps=32)
        for streak, expected in ((0, False), (31, False), (32, True)):
            event = dict(status='accepted', trajectory_step=40, neutral_streak=streak, cost=dict(gates=37))
            self.assertEqual(immediate_terminal(event, config, 40), expected)
            self.assertFalse(immediate_terminal(event, dict(config, neutral_stop_steps=0), 40))
