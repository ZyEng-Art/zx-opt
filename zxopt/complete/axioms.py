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


NAMES = ("S", "I", "CP", "B", "K", "EU", "H", "SUP", "E", "C", "BW")
SOURCE = "https://arxiv.org/abs/1903.06035v3"


@dataclass(frozen=True)
class Axiom:
    name: str
    parameters: dict
    left: Diagram
    right: Diagram

    def record(self):
        return dict(name=self.name, parameters={k: str(v) if isinstance(v, F) else v
                                               for k, v in self.parameters.items()})

    @classmethod
    def from_record(cls, record):
        parameters = dict(record["parameters"])
        for k in ("alpha", "beta", "gamma"):
            if k in parameters:
                parameters[k] = F(parameters[k])
        return axiom(record["name"], **parameters)


def axiom(name, *, color="Z", **parameters):
    if name not in NAMES or color not in ("Z", "X"):
        raise ValueError("Unknown axiom or color")
    allowed = dict(S={"alpha", "beta", "left_legs", "right_legs", "links"},
                   I=set(), CP=set(), B=set(), K={"alpha"}, EU=set(),
                   H={"alpha", "arity"}, SUP={"alpha"}, E=set(),
                   C={"alpha", "beta", "gamma"}, BW=set())[name]
    if set(parameters) - allowed:
        raise ValueError("Unexpected axiom parameters")
    p = dict(parameters)
    for k in ("alpha", "beta", "gamma"):
        if k in allowed:
            p[k] = quarter(p.get(k, F(1, 4)))
    defaults = dict(left_legs=1, right_legs=1, links=1, arity=2)
    for k in allowed & defaults.keys():
        value = p.get(k, defaults[k])
        if not isinstance(value, int) or value < (1 if k == "links" else 0):
            raise ValueError("Arities are nonnegative; spider links are positive")
        p[k] = value
    z, x = color, "X" if color == "Z" else "Z"
    l, r = Diagram(), Diagram()

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
    return Axiom(name, dict(color=color, **p), l.validate(), r.validate())


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
