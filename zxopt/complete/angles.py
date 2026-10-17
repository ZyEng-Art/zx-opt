"""Real phase expressions in units of pi, with a safe JSON expression language.

The two Mean functions are total real functions of four real phase arguments.
They implement the constructive polar solution of JPV Figure 3, including
arg(0)=0. Expressions are never parsed with eval/sympify on untrusted strings.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from functools import lru_cache
import math
import re

import sympy as s
from sympy.core.evalf import prec_to_dps


@lru_cache(maxsize=4096)
def mean_z(a, b, t1, t2):
    return (s.exp(s.I*s.pi*t1)*s.cos(s.pi*a) +
            s.exp(s.I*s.pi*t2)*s.cos(s.pi*b))/2


class MeanArg(s.Function):
    nargs = 4
    is_real = True

    @classmethod
    def eval(cls, a, b, t1, t2):
        if any(v.is_real is not True for v in (a, b, t1, t2)):
            raise ValueError("MeanArg requires four real phase arguments")
        z = mean_z(a, b, t1, t2)
        if z.is_zero is True or z.is_positive is True:
            return s.Integer(0)
        if z.is_negative is True:
            return s.Integer(1)

    def _eval_evalf(self, prec):
        digits = prec_to_dps(prec)+15
        z = mean_z(*self.args).evalf(digits)
        return (s.arg(z)/s.pi if z != 0 else s.Integer(0)).evalf(digits)


class MeanAcos(s.Function):
    nargs = 4
    is_real = True

    @classmethod
    def eval(cls, a, b, t1, t2):
        if any(v.is_real is not True for v in (a, b, t1, t2)):
            raise ValueError("MeanAcos requires four real phase arguments")
        z = mean_z(a, b, t1, t2)
        if z.is_zero is True:
            return s.Rational(1, 2)
        r = s.Abs(z)
        if r.is_Rational and 0 <= r <= 1:
            return s.acos(r)/s.pi

    def _eval_evalf(self, prec):
        digits = prec_to_dps(prec)+15
        z = mean_z(*self.args).evalf(digits)
        r = s.Abs(z).evalf(digits)
        # The unit-disc bound follows from the real input angles. Never turn
        # a genuine out-of-domain value into a valid constraint by clipping.
        if r > 1:
            if (r-1).evalf(digits) > s.Float(10)**(-digits+5):
                raise ValueError("Constructive A solution escaped the unit disc")
            r = s.Integer(1)  # rounding at the working precision only
        return (s.acos(r)/s.pi).evalf(digits)


def expr(value):
    if isinstance(value, Phase):
        return value.expression
    if isinstance(value, s.Expr):
        return value
    if isinstance(value, float):
        raise ValueError("Use exact rational or symbolic phases, not float")
    q = F(value)
    return s.Rational(q.numerator, q.denominator)


def phase(value):
    e = s.expand_mul(expr(value))
    if e.is_real is not True or e.has(s.Float, s.nan, s.zoo, s.oo, -s.oo):
        raise ValueError("A phase must be a provably real exact expression")
    constant, rest = e.as_coeff_Add()
    if constant.is_Rational:
        e = rest + constant % 2
    if e.is_Rational:
        return F(int(s.numer(e)), int(s.denom(e)))
    return Phase(e)


@dataclass(frozen=True, eq=False)
class Phase:
    expression: object

    def __post_init__(self):
        if not isinstance(self.expression, s.Expr) or self.expression.is_real is not True:
            raise ValueError("Phase expects a real symbolic expression")
        if self.expression.has(s.Float, s.nan, s.zoo, s.oo, -s.oo):
            raise ValueError("Phase expressions must be exact and finite")

    def __str__(self):
        return str(self.expression)

    def __bool__(self):
        return self.expression != 0

    def __float__(self):
        if self.expression.free_symbols:
            raise ValueError("Bind symbolic angles before numeric scoring")
        result = float(self.expression.evalf(25))
        if not math.isfinite(result):
            raise ValueError("Nonfinite phase evaluation")
        return result

    def __eq__(self, other):
        try:
            return self.expression == expr(other)
        except (TypeError, ValueError):
            return False

    def __hash__(self):
        return hash(self.expression)

    def __add__(self, other): return phase(self.expression+expr(other))
    def __radd__(self, other): return self+other
    def __neg__(self): return phase(-self.expression)
    def __sub__(self, other): return self+-phase(other)
    def __rsub__(self, other): return phase(other)+-self
    def __mul__(self, other): return phase(self.expression*expr(other))
    def __rmul__(self, other): return self*other
    def __truediv__(self, other): return phase(self.expression/expr(other))
    def __mod__(self, modulus):
        if modulus != 2:
            raise ValueError("Phases are normalized modulo 2 in units of pi")
        return phase(self.expression)


def symbol(name):
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name):
        raise ValueError("Invalid real phase symbol")
    return phase(s.Symbol(name, real=True))


OPS = {"add": s.Add, "mul": s.Mul, "pow": s.Pow, "cos": s.cos,
       "sin": s.sin, "exp": s.exp, "acos": s.acos, "abs": s.Abs,
       "arg": s.arg, "mean_arg": MeanArg, "mean_acos": MeanAcos}
OP_NAMES = {fn: name for name, fn in OPS.items()}


def encode(e):
    if e.is_Rational:
        return ["q", str(e.p), str(e.q)]
    if e == s.pi: return ["pi"]
    if e == s.I: return ["i"]
    if isinstance(e, s.Symbol):
        if e.is_real is not True:
            raise ValueError("Only real symbols are supported")
        return ["symbol", str(e)]
    if e.func not in OP_NAMES:
        raise ValueError("Unsupported expression operator: "+str(e.func))
    return [OP_NAMES[e.func], *[encode(a) for a in e.args]]


def decode(row, depth=0):
    if not isinstance(row, list) or not row:
        raise ValueError("Invalid phase expression record")
    op, *args = row
    if op == "q" and len(args) == 2:
        return s.Rational(int(args[0]), int(args[1]))
    if op == "pi" and not args: return s.pi
    if op == "i" and not args: return s.I
    if op == "symbol" and len(args) == 1:
        return expr(symbol(args[0]))
    if op not in OPS:
        raise ValueError("Unknown phase expression operator")
    return OPS[op](*(decode(a, depth+1) for a in args))


def record(value):
    return {"expression": encode(value.expression)} if isinstance(value, Phase) else str(value)


def restore(value):
    if isinstance(value, dict):
        if set(value) != {"expression"}:
            raise ValueError("Invalid phase expression fields")
        return phase(decode(value["expression"]))
    return F(value)


def numeric(value, digits=40):
    if not isinstance(value, Phase):
        return F(value)
    if value.expression.free_symbols:
        raise ValueError("Cannot score an unbound real phase symbol")
    e = value.expression.evalf(digits)
    if e.is_real is not True or not e.is_finite:
        raise ValueError("Invalid numeric expression evaluation")
    return F(str(e)) % 2


def solve_a(alpha, beta, theta1, theta2):
    args = tuple(expr(phase(v)) for v in (alpha, beta, theta1, theta2))
    return phase(MeanArg(*args)), phase(MeanAcos(*args))


class SideConditionError(ValueError):
    def __init__(self, status):
        self.status = status
        super().__init__("A side condition is "+status+"; numeric residuals are insufficient")


def a_evidence(alpha, beta, theta1, theta2, theta3, gamma):
    """Verify a side-condition certificate; a small residual is not a proof."""
    a, b, t1, t2, t3, g = map(phase, (alpha, beta, theta1, theta2, theta3, gamma))
    if a == b == g and t1 == t2 == t3:
        return "diagonal"
    if a == b == 0 and t1 == phase(t3+g) and t2 == phase(t3-g):
        return "symmetric"
    if (t3, g) == solve_a(a, b, t1, t2):
        return "constructive-polar"
    residual = (2*s.exp(s.I*s.pi*expr(t3))*s.cos(s.pi*expr(g)) -
                2*mean_z(*(expr(v) for v in (a, b, t1, t2))))
    if s.count_ops(residual) <= 80:
        residual = s.simplify(residual)
        if residual == 0:
            return "symbolic-identity"
    raise SideConditionError("disproved" if residual.is_zero is False else "unknown")
