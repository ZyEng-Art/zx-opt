"""Undiscounted credit preserves endpoint gate gains through temporary growth."""
import math
import unittest

import torch

from zxopt.anchor_ppo import gae


class ReturnCreditTest(unittest.TestCase):
    def test_monte_carlo_returns_telescope_despite_temporary_gate_growth(self):
        costs = [40,59]+[59]*20+[38]
        rewards = [(a-b)/10 for a,b in zip(costs,costs[1:])]
        values = [math.sin(i)/3 for i in range(len(rewards))]
        dones = [False]*(len(rewards)-1)+[True]
        _, actual = gae(rewards,dones,values,999.,gamma=1.,lam=1.)
        expected = torch.tensor([(costs[i]-costs[-1])/10 for i in range(len(rewards))])
        torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
        self.assertAlmostEqual(float(actual[0]),.2,places=6)
        _, discounted = gae(rewards,dones,[0.]*len(rewards),999.,gamma=.99,lam=1.)
        self.assertLess(float(discounted[0]),0.)

    def test_terminal_boundaries_and_collection_cut_bootstrap_are_separate(self):
        _, actual = gae([.1,-.2,.4,-.3,.2],[False,True,False,False,True],
                        [.7,-.6,.5,-.4,.3],100.,gamma=1.,lam=1.)
        torch.testing.assert_close(actual,torch.tensor([-.1,-.2,.3,-.1,.2]),rtol=1e-6,atol=1e-6)
        _, pending = gae([.1,.2],[False,False],[.7,-.6],3.,gamma=1.,lam=1.)
        torch.testing.assert_close(pending,torch.tensor([3.3,3.2]),rtol=1e-6,atol=1e-6)

    def test_explicit_default_credit_is_bitwise_identical(self):
        args=([.1,-.3,0.,.5],[False,False,True,False],[.2,.1,-.2,.4],.7)
        old=gae(*args);new=gae(*args,gamma=.99,lam=.95)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(old,new)))
