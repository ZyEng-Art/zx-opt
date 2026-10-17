"""JPV 1903.06035v3, Figures 1/2 and Theorem 1.1, scalar-accurate.

The Clifford+T presentation excludes IV/ZO, as specified in Figure 2. Colours
may be exchanged. Wire bending/upside-down instances are supplied by arbitrary
port embeddings, rather than by imposing a circuit direction on a pattern.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import product

from .diagram import Diagram
from .ring import quarter
from .angles import Phase, phase, record as phase_record, restore as restore_phase, solve_a, a_evidence


NAMES = ("S", "I", "CP", "B", "K", "EU", "H", "SUP", "E", "C", "BW")
GENERAL_NAMES = NAMES + ("A",)
SOURCE = "https://arxiv.org/abs/1903.06035v3"


@dataclass(frozen=True)
class Axiom:
    name: str
    parameters: dict
    left: Diagram
    right: Diagram
    domain: str = "clifford_t"
    evidence: str = ""

    def record(self):
        row = dict(name=self.name, parameters={k: phase_record(v) if isinstance(v, (F, Phase)) else v
                                              for k, v in self.parameters.items()})
        if self.domain != "clifford_t":
            row["domain"] = self.domain
        if self.evidence:
            row["constraint_evidence"] = self.evidence
        return row

    @classmethod
    def from_record(cls, record):
        parameters = dict(record["parameters"])
        for k in ("alpha", "beta", "gamma", "theta1", "theta2", "theta3"):
            if k in parameters:
                parameters[k] = restore_phase(parameters[k])
        rule = axiom(record["name"], domain=record.get("domain", "clifford_t"), **parameters)
        if record.get("constraint_evidence", "") != rule.evidence:
            raise ValueError("Axiom side-condition evidence changed on reconstruction")
        return rule


def axiom(name, *, color="Z", domain="clifford_t", **parameters):
    if domain not in ("clifford_t", "general"):
        raise ValueError("Unknown phase domain")
    if name not in (GENERAL_NAMES if domain == "general" else NAMES) or color not in ("Z", "X"):
        raise ValueError("Unknown axiom or color")
    allowed = dict(S={"alpha", "beta", "left_legs", "right_legs", "links"},
                   I=set(), CP=set(), B=set(), K={"alpha"}, EU=set(),
                   H={"alpha", "arity"}, SUP={"alpha"}, E=set(),
                   C={"alpha", "beta", "gamma"}, BW=set(),
                   A={"alpha", "beta", "gamma", "theta1", "theta2", "theta3"})[name]
    if set(parameters) - allowed:
        raise ValueError("Unexpected axiom parameters")
    p = dict(parameters)
    evidence = ""
    if name == "A":
        for k in ("alpha", "beta", "theta1", "theta2"):
            p[k] = phase(p.get(k, F(1, 3) if k == "beta" else F(0)))
        if ("theta3" in p) != ("gamma" in p):
            raise ValueError("Supply both theta3/gamma, or let the constructive solver choose both")
        if "theta3" not in p:
            p["theta3"], p["gamma"] = solve_a(*(p[k] for k in ("alpha", "beta", "theta1", "theta2")))
        else:
            p["theta3"], p["gamma"] = phase(p["theta3"]), phase(p["gamma"])
        evidence = a_evidence(*(p[k] for k in ("alpha", "beta", "theta1", "theta2", "theta3", "gamma")))
    for k in ("alpha", "beta", "gamma"):
        if k in allowed:
            p[k] = (phase if domain == "general" else quarter)(p.get(k, F(1, 4)))
    defaults = dict(left_legs=1, right_legs=1, links=1, arity=2)
    for k in allowed & defaults.keys():
        value = p.get(k, defaults[k])
        if not isinstance(value, int) or value < (1 if k == "links" else 0):
            raise ValueError("Arities are nonnegative; spider links are positive")
        p[k] = value
    z, x = color, "X" if color == "Z" else "Z"
    l, r = Diagram(allow_arbitrary=domain == "general"), Diagram(allow_arbitrary=domain == "general")

    def scalar(d, phase=0):
        a, b = d.node(x), d.node(z, phase)
        d.edge(a, b)

    def wire(d):
        a, b = d.boundary(), d.boundary()
        d.edge(a, b)

    def chain(d, specs):
        nodes = [d.node(k, phase) for k, phase in specs]
        for a, b in zip(nodes, nodes[1:]):
            d.edge(a, b)
        d.leg(nodes[0]); d.leg(nodes[-1])
        return nodes

    if name == "S":
        a, b = l.node(z, p["alpha"]), l.node(z, p["beta"])
        for _ in range(p["links"]):
            l.edge(a, b)
        c = r.node(z, p["alpha"] + p["beta"])
        for v, count in ((a, p["left_legs"]), (b, p["right_legs"])):
            for _ in range(count):
                l.leg(v); r.leg(c)
    elif name == "I":
        v = l.node(z); l.leg(v); l.leg(v)
        wire(r)
    elif name == "CP":
        scalar(l)
        a, b = l.node(x), l.node(z)
        l.edge(a, b); l.leg(b); l.leg(b)
        r.leg(r.node(x)); r.leg(r.node(x))
    elif name == "B":
        scalar(l)
        top = [l.node(z), l.node(z)]
        bottom = [l.node(x), l.node(x)]
        for a in top:
            for b in bottom:
                l.edge(a, b)
        for v in top + bottom:
            l.leg(v)
        a, b = r.node(x), r.node(z)
        r.edge(a, b)
        r.leg(a); r.leg(a); r.leg(b); r.leg(b)
    elif name == "K":
        scalar(l)
        chain(l, [(x, p["alpha"]), (z, 1)])
        # The scalar on the RHS is an alpha-phase X leaf joined to a pi Z leaf.
        a, b = r.node(x, p["alpha"]), r.node(z, 1)
        r.edge(a, b)
        chain(r, [(z, 1), (x, -p["alpha"])])
    elif name == "EU":
        v = l.node("H"); l.leg(v); l.leg(v)
        nodes = chain(r, [(z, F(1, 2)), (x, 0), (z, F(1, 2))])
        r.edge(nodes[1], r.node(z, F(-1, 2)))
    elif name == "H":
        a, b = l.node(x, p["alpha"]), r.node(z, p["alpha"])
        for _ in range(p["arity"]):
            h = l.node("H"); l.edge(a, h); l.leg(h); r.leg(b)
    elif name == "SUP":
        a, b, c = l.node(z, p["alpha"]), l.node(z, p["alpha"] + 1), l.node(x)
        l.edge(a, c); l.edge(b, c); l.leg(c)
        a, b = r.node(z, 2*p["alpha"] + 1), r.node(x)
        r.edge(a, b); r.edge(a, b); r.leg(b)
    elif name == "E":
        a, b = l.node(z, F(1, 4)), l.node(x, F(-1, 4))
        l.edge(a, b)
    elif name == "C":
        def controlled(d, swapped):
            # Ordered interface: three top wires left-to-right, then bottom.
            top = [d.boundary() for _ in range(3)]
            middle = d.node(z)
            hub = d.node(z)
            d.edge(top[1], middle)
            stems = {side: d.node(z, p["alpha"] if side == (2 if swapped else 0)
                                  else p["beta"]) for side in (0, 2)}
            for side in (0, 2):
                alpha_side = (side == (2 if swapped else 0))
                phase = p["alpha"] if alpha_side else p["beta"]
                stem = stems[side]
                control = d.node(x, 0 if alpha_side else 1)
                leaf = d.node(z, phase)
                d.edge(stem, control); d.edge(control, leaf); d.edge(control, hub)
                if alpha_side:
                    sign = 1 if swapped else -1
                    upper = d.node(x, sign*p["gamma"])
                    lower = d.node(x, -sign*p["gamma"])
                    d.edge(top[side], upper); d.edge(upper, stem)
                    d.edge(upper, middle); d.edge(middle, lower); d.edge(lower, stem)
                    d.edge(lower, stems[2-side])
                else:
                    d.edge(top[side], stem)
            d.leg(hub)
        controlled(l, False); controlled(r, True)
    elif name == "A":
        # JPV Figure 3, including every scalar-carrying leaf. Both sides have
        # one ordered open port; colour duality follows tensor conjugation by H.
        sides = []
        for a, b, theta in ((-p["alpha"], p["alpha"], p["theta1"]),
                            (p["beta"], -p["beta"], p["theta2"])):
            tip = l.node(x)
            l.edge(tip, l.node(z, a)); l.edge(tip, l.node(z, b))
            middle = l.node(z, theta); l.edge(tip, middle)
            join = l.node(x); l.edge(middle, join); sides.append(join)
        hub = l.node(z); l.edge(hub, l.node(x, F(1, 2)))
        bottom = l.node(x)
        for join in sides:
            l.edge(join, hub)
            v = l.node(z, F(1, 4)); l.edge(join, v); l.edge(v, bottom)
        l.leg(bottom)
        root = r.node(z, p["theta3"])
        for a, b in ((-p["gamma"], p["gamma"]), (F(1, 4), F(1, 4))):
            tip = r.node(x); r.edge(root, tip)
            r.edge(tip, r.node(z, a)); r.edge(tip, r.node(z, b))
        r.leg(root)
    elif name == "BW":
        def leaf(d, v):
            d.edge(v, d.node(z, F(1, 4)))
        top, bottom = l.node(z), l.node(z)
        a, b, c = l.node(x), l.node(z, F(-1, 2)), l.node(x)
        for u, v in ((top, a), (a, b), (b, c), (c, bottom)):
            l.edge(u, v)
        leaf(l, a); leaf(l, c)
        for v in (top, bottom):
            u = l.node(x); l.edge(v, u); leaf(l, u); leaf(l, u)
        l.leg(top); l.leg(bottom)
        top, bottom = r.node(z), r.node(x, F(1, 2))
        a, b = r.node(x, 1), r.node(z, F(1, 4))
        r.edge(top, a); r.edge(a, b); r.edge(b, bottom); leaf(r, a)
        u = r.node(x, 1); r.edge(top, u); leaf(r, u); leaf(r, u)
        r.leg(top); r.leg(bottom)
    return Axiom(name, dict(color=color, **p), l.validate(), r.validate(), domain, evidence)


def instances_at(rank):
    """Finite stages whose union covers ALL pi/4 axiom instances fairly.

    rank = external legs + parallel fusion links - 1 for S; arity for H;
    fixed-arity rules occur at rank zero. No hidden vertex/candidate cutoff.
    The stages are disjoint; phase parameters cover all eight values modulo 2pi.
    """
    if not isinstance(rank, int) or rank < 0:
        raise ValueError("Rank must be a nonnegative integer")
    phases = tuple(F(k, 4) for k in range(8))
    for color in ("Z", "X"):
        for a, b in product(phases, repeat=2):
            for left in range(rank+1):
                for right in range(rank-left+1):
                    yield axiom("S", color=color, alpha=a, beta=b, left_legs=left,
                                right_legs=right, links=rank-left-right+1)
        for a in phases:
            yield axiom("H", color=color, alpha=a, arity=rank)
        if rank == 0:
            for name in ("I", "CP", "B", "EU", "E", "BW"):
                yield axiom(name, color=color)
            for a in phases:
                yield axiom("K", color=color, alpha=a)
                yield axiom("SUP", color=color, alpha=a)
            for a, b, c in product(phases, repeat=3):
                yield axiom("C", color=color, alpha=a, beta=b, gamma=c)
