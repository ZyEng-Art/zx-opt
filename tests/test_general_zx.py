"""General phase schemata, certified nonlinear A, and the actual policy path."""
from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import sympy as s
import torch

from zxopt.complete import (GENERAL_NAMES, Diagram, RuleAction, axiom, actions,
                            exact_tensor, tensor_product, join_ports, symbol)
from zxopt.complete.angles import Phase, phase, record, restore, numeric, solve_a
from zxopt.complete.adapter import to_pyzx, from_pyzx
from zxopt.general_kernel import GeneralKernel, Views
from zxopt.general_policy import GeneralAgent, observation, batch
from zxopt.jpv_actions import Proposals, cancel_h_pairs, replay_h_pairs
from zxopt.legacy import environment
from zxopt.verify import graph_matrix
from zxopt.general_search import ppo_update
from test_complete import independent_tensor


def matrix(d):
    return graph_matrix(to_pyzx(d, len(d.ports)//2, evaluate_expressions=True))


def unitary_a_host(theta):
    """Embed Figure 3 RHS in a one-wire Z phase with exact CT compensation."""
    import pyzx as zx
    r = axiom("A", domain="general", alpha=F(1, 4), beta=F(1, 4),
              theta1=theta, theta2=theta, theta3=theta, gamma=F(1, 4))
    wire = Diagram(allow_arbitrary=True)
    v = wire.node("Z")
    for _ in range(3): wire.leg(v)
    combined = tensor_product(r.right, wire)
    combined = join_ports(combined, [(combined.ports[0], combined.ports[-1])])
    scalar = zx.Graph(); scalar.scalar.add_power(-1); scalar.scalar.add_phase(F(-1, 4))
    return tensor_product(combined, from_pyzx(scalar))


class GeneralZXTest(unittest.TestCase):
    def test_phase_expressions_roundtrip_and_no_implicit_numeric_export(self):
        a = symbol("alpha")
        self.assertEqual(phase(a+2), a)
        self.assertEqual(phase(a-a), F(0))
        self.assertEqual(restore(json.loads(json.dumps(record(a/F(3)+F(1, 8))))), a/F(3)+F(1, 8))
        with self.assertRaises(ValueError): numeric(a)
        with self.assertRaises(ValueError): restore({"expression": ["eval", "__import__('os')"]})
        with self.assertRaises(ValueError):
            restore({"expression": ["mean_acos", ["i"], ["q","0","1"], ["q","0","1"], ["q","0","1"]]})
        rule = axiom("A", domain="general")
        gamma = rule.parameters["gamma"]
        self.assertIsInstance(gamma, Phase)
        self.assertAlmostEqual(float(gamma), np.arccos(.75)/np.pi)
        self.assertEqual(restore(json.loads(json.dumps(record(gamma)))), gamma)
        with self.assertRaises(ValueError): exact_tensor(rule.right)
        with self.assertRaises(ValueError): to_pyzx(rule.right, 0)
        np.testing.assert_allclose(matrix(rule.left), matrix(rule.right), atol=2e-12)

    def test_general_linear_rules_accept_symbolic_parameters_and_replay(self):
        a, b, g = symbol("a"), symbol("b"), symbol("g")
        for name in ("S", "H", "K", "SUP", "C"):
            kwargs = dict(alpha=a)
            if name == "C": kwargs.update(beta=b, gamma=g)
            rule = axiom(name, domain="general", **kwargs)
            for direction in ("forward", "backward"):
                source, target = ((rule.left, rule.right) if direction == "forward"
                                  else (rule.right, rule.left))
                action = next(actions(source, rule, direction))
                child = RuleAction.from_record(json.loads(json.dumps(action.record()))).apply(source)
                self.assertTrue(child.isomorphic(target), (name, direction))
        strict = axiom("S", alpha=F(0), beta=F(0)).right
        general = axiom("S", domain="general", alpha=F(1,3), beta=F(5,3))
        child = next(actions(strict,general,"backward")).apply(strict)
        self.assertTrue(child.allow_arbitrary)
        np.testing.assert_allclose(matrix(child),matrix(strict),atol=2e-12)

    def test_a_fig3_scalar_and_color_against_independent_incidence_tensors(self):
        for color, a, b, t1, t2 in product(
                ("Z", "X"), (F(0), F(1, 3)), (F(1, 5), F(1, 2)),
                (F(0), F(2, 7)), (F(1, 8),)):
            rule = axiom("A", domain="general", color=color, alpha=a, beta=b, theta1=t1, theta2=t2)
            left, right = independent_tensor(rule.left), independent_tensor(rule.right)
            np.testing.assert_allclose(left, right, atol=4e-12)
            theta, gamma = rule.parameters["theta3"], rule.parameters["gamma"]
            expected = np.array([1+1j, 2*np.exp(1j*np.pi*(float(theta)+.25))*np.cos(np.pi*float(gamma))])
            if color == "X":
                expected = np.array([[1, 1], [1, -1]]) @ expected/np.sqrt(2)
            np.testing.assert_allclose(right, expected, atol=4e-12)
            np.testing.assert_allclose(matrix(rule.right).flatten(), right, atol=4e-12)

    def test_a_zero_cancellation_unit_disc_boundary_and_exact_rejection(self):
        for a,b,t1,t2 in ((F(1,2),F(1,2),F(0),F(0)),
                          (F(0),F(0),F(0),F(1)),
                          (F(0),F(0),F(0),F(0))):
            rule = axiom("A", domain="general", alpha=a, beta=b, theta1=t1, theta2=t2)
            np.testing.assert_allclose(independent_tensor(rule.left), independent_tensor(rule.right), atol=3e-12)
            theta,gamma = solve_a(a,b,t1,t2)
            if a == F(1,2) or t2 == 1:
                self.assertEqual((theta,gamma), (F(0),F(1,2)))
        with self.assertRaises(ValueError):
            axiom("A", domain="general", alpha=F(0), beta=F(0), theta1=F(0), theta2=F(0),
                  theta3=F(1,10**15), gamma=F(0))
        rule = axiom("A", domain="general")
        action = next(actions(rule.left, rule))
        row = action.record(); row["axiom"]["constraint_evidence"] = "unchecked"
        with self.assertRaises(ValueError): RuleAction.from_record(row)
        row = action.record(); row["axiom"]["parameters"]["gamma"] = "0"
        with self.assertRaises(ValueError): RuleAction.from_record(row)

    def test_all_general_families_and_both_directions_proposed(self):
        proposals = Proposals(seed=29, cap=128, matching_work=16384, general=True)
        for name, color in product(GENERAL_NAMES, ("Z", "X")):
            rule = axiom(name, domain="general", color=color)
            for direction in ("forward", "backward"):
                source = rule.left if direction == "forward" else rule.right
                relevant = [a for a in proposals.build(source)
                            if a.axiom.name == name and a.direction == direction]
                self.assertTrue(relevant, (name,color,direction))
                for action in relevant[:2]:
                    child = RuleAction.from_record(json.loads(json.dumps(action.record()))).apply(source)
                    child, receipts = cancel_h_pairs(child)
                    self.assertEqual(replay_h_pairs(action.apply(source), receipts).digest(), child.digest())
                    np.testing.assert_allclose(matrix(child), matrix(source), atol=4e-11)

    def test_nonquarter_phase_inference_and_both_sup_roots(self):
        proposals = Proposals(cap=128, matching_work=16384, general=True)
        for name in ("K", "SUP", "C", "H"):
            kwargs = dict(alpha=F(1,3))
            if name == "C": kwargs.update(beta=F(1,5),gamma=F(2,7))
            rule = axiom(name, domain="general", **kwargs)
            for direction in ("forward","backward"):
                source = rule.left if direction == "forward" else rule.right
                aa = [a for a in proposals.build(source) if a.axiom.name==name and a.direction==direction]
                self.assertTrue(aa, (name,direction))
                for a in aa: np.testing.assert_allclose(matrix(a.apply(source)),matrix(source),atol=4e-11)
                if name=="SUP" and direction=="backward":
                    self.assertEqual({a.axiom.parameters["alpha"] for a in aa}, {F(1,3),F(4,3)})

    def test_nonlinear_a_inside_a_real_unitary_and_numeric_view_isolation(self):
        host = unitary_a_host(F(1,3))
        reference = np.diag([1.,np.exp(1j*np.pi/3)])
        np.testing.assert_allclose(matrix(host), reference, atol=3e-12)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"rz.qasm"
            p.write_text('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nrz(pi/3) q[0];\n')
            kernel=GeneralKernel(environment(p),candidate_cap=128,matching_work=16384,general_zx=True)
            views=Views(1,reference); view=views.get(host)
            aa=kernel.prepare(host,view,view.graph.num_vertices()+64)
            back=next(a for a in aa if a.family=="A:backward")
            left,_=kernel.step(host,view,back)
            lv=views.get(left)
            aa=kernel.prepare(left,lv,view.graph.num_vertices()+64)
            forward=next(a for a in aa if a.family=="A:forward")
            child,_=kernel.step(left,lv,forward)
            self.assertTrue(any(isinstance(n.phase,Phase) for n in child.nodes.values()))
            digest=child.digest(); checked=views.get(child)
            self.assertEqual(child.digest(),digest)
            self.assertLess(checked.error,2e-8)
            proposals=kernel.prepare(child,checked,view.graph.num_vertices()+64)
            self.assertTrue(proposals)
            self.assertTrue(all(a.mode=="jpv" for a in proposals))

    def test_a_action_density_and_actual_ppo_gradient(self):
        torch.set_num_threads(1)
        host=unitary_a_host(F(1,3)); proposals=Proposals(cap=128,matching_work=16384,general=True)
        actions_ = proposals.build(host)
        from zxopt.general_kernel import HybridAction
        available=[]
        for a in actions_:
            q=a.axiom.parameters
            available.append(HybridAction("jpv",a.axiom.name+":"+a.direction,a,None,None,
                tuple(w for _,w in a.match.vertices),tuple(q.get(k,F(0)) for k in ("alpha","beta","gamma")),
                (0.,0.,0.),0,a.direction))
        slot=next(i for i,a in enumerate(available) if a.family=="A:backward")
        obs=observation(host,None,available,[0.,0.,0.,0.,1.,1.,0.,0.])
        net=GeneralAgent(general_zx=True); packed=batch([obs])
        _,logs,_,values,_,_=net.get_action_and_value(packed,action=torch.tensor([slot]))
        logits=net.actor(packed[0]).flatten()[packed[0].y!=-1]
        torch.testing.assert_close(logs[0],logits.log_softmax(0)[slot])
        rollout=[(obs,slot,float(logs[0].detach()),float(values[0].detach()),.1 if i%2 else -.1,i==7)
                 for i in range(8)]
        update=ppo_update(net,torch.optim.Adam(net.parameters(),lr=.0003),rollout,0.,
                          dict(epochs=1,minibatch=4),time.perf_counter()+30)
        self.assertGreater(update["actor_parameter_l2_delta"],0.)
        self.assertTrue(np.isfinite(update["mean_loss"]))


if __name__=="__main__": unittest.main()
