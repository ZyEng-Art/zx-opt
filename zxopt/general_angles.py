"""Bounded general-phase instances inferred from topology and affine labels.

The trusted registry checks every resulting rule again. No numeric phase
comparison is used. Nonlinear A output parameters are constructed analytically.
"""
from fractions import Fraction as F
from functools import lru_cache

import sympy as s

from .complete import axiom
from .complete.angles import phase


def source_rule(name, direction, color, arity, values):
    params = dict(values)
    if name == "H":
        params["arity"] = arity
    if name == "A" and direction == "backward":
        gamma, theta = params["gamma"], params["theta3"]
        params.update(alpha=gamma, beta=gamma, theta1=theta, theta2=theta)
    return axiom(name, color=color, domain="general", **params)


@lru_cache(maxsize=256)
def compiled(name, direction, color, arity=0):
    names = dict(K=("alpha",), SUP=("alpha",), C=("alpha", "beta", "gamma"),
                 H=("alpha",)).get(name, ())
    if name == "A":
        names = (("alpha", "beta", "theta1", "theta2") if direction == "forward"
                 else ("gamma", "theta3"))
    zero = dict.fromkeys(names, F(0))
    rule = source_rule(name, direction, color, arity, zero)
    source = rule.left if direction == "forward" else rule.right
    internal = tuple(v for v in source.nodes if v not in source.ports)
    constants = tuple(source.nodes[v].phase for v in internal)
    columns = []
    for k in names:
        values = dict(zero); values[k] = F(1, 4)
        r = source_rule(name, direction, color, arity, values)
        side = r.left if direction == "forward" else r.right
        column = [int((side.nodes[v].phase-c)*4) % 8 for v, c in zip(internal, constants)]
        columns.append([a-8 if a > 4 else a for a in column])
    if names:
        matrix = s.Matrix(len(internal), len(names), lambda i, j: columns[j][i])
        rows = tuple(matrix.T.rref()[1])
        if len(rows) != len(names):
            raise ValueError("General-phase source has undetermined parameters")
        inverse = matrix[list(rows), :].inv()
        inverse = tuple(tuple(F(v) for v in inverse.row(i)) for i in range(len(names)))
    else:
        rows = inverse = ()
    return source, internal, constants, names, rows, inverse


def inferred_rules(graph, match, name, direction, color, arity=0):
    source, internal, constants, names, rows, inverse = compiled(name, direction, color, arity)
    mapping = dict(match.vertices)
    rhs = tuple(graph.nodes[mapping[v]].phase-c for v, c in zip(internal, constants))
    solved = [phase(sum((q*rhs[j] for q, j in zip(row, rows)), F(0))) for row in inverse]
    # SUP backward solves 2*alpha+1=phase modulo 2; retain both roots.
    shifts = (F(0), F(1)) if name == "SUP" and direction == "backward" else (F(0),)
    seen = set()
    for shift in shifts:
        values = {k: phase(v+shift) for k, v in zip(names, solved)}
        rule = source_rule(name, direction, color, arity, values)
        side = rule.left if direction == "forward" else rule.right
        if any(side.nodes[v] != graph.nodes[mapping[v]] for v in internal):
            continue
        signature = str(rule.record())
        if signature not in seen:
            seen.add(signature); yield rule
        if name == "A" and direction == "backward":
            # A second exact inverse family: exp(i(t+g))+exp(i(t-g))
            # = 2 exp(it) cos(g). No continuous-angle regression is required.
            g, t = values["gamma"], values["theta3"]
            alternate = axiom("A", color=color, domain="general", alpha=F(0), beta=F(0),
                              theta1=phase(t+g), theta2=phase(t-g), theta3=t, gamma=g)
            signature = str(alternate.record())
            if signature not in seen:
                seen.add(signature); yield alternate
