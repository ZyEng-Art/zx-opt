"""Exact dyadic arithmetic in Q[zeta]/(zeta**4 + 1), zeta = exp(i*pi/4)."""
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class C8:
    coefficients: tuple = (Fraction(0),) * 4

    def __post_init__(self):
        c = tuple(Fraction(v) for v in self.coefficients)
        if len(c) != 4 or any(v.denominator & (v.denominator - 1) for v in c):
            raise ValueError("Clifford+T coefficients must be four dyadic rationals")
        object.__setattr__(self, "coefficients", c)

    @classmethod
    def of(cls, value):
        return value if isinstance(value, cls) else cls((Fraction(value), 0, 0, 0))

    @classmethod
    def phase(cls, quarter_turns):
        k = int(quarter_turns) % 8
        c = [0] * 4
        c[k % 4] = 1 if k < 4 else -1
        return cls(tuple(c))

    def __add__(self, other):
        b = C8.of(other)
        return C8(tuple(x + y for x, y in zip(self.coefficients, b.coefficients)))

    __radd__ = __add__

    def __neg__(self):
        return C8(tuple(-x for x in self.coefficients))

    def __sub__(self, other):
        return self + -C8.of(other)

    def __mul__(self, other):
        b = C8.of(other)
        c = [Fraction(0)] * 4
        for i, x in enumerate(self.coefficients):
            if not x:
                continue
            for j, y in enumerate(b.coefficients):
                if y:
                    c[(i+j) % 4] += x * y * (1 if i+j < 4 else -1)
        return C8(tuple(c))

    __rmul__ = __mul__

    def __bool__(self):
        return any(self.coefficients)

    def __complex__(self):
        import cmath
        return sum(float(c) * cmath.exp(1j * k * cmath.pi / 4)
                   for k, c in enumerate(self.coefficients))

    def record(self):
        return [str(c) for c in self.coefficients]


ZERO = C8()
ONE = C8.of(1)
SQRT2 = C8((0, 1, 0, -1))
INV_SQRT2 = C8((0, Fraction(1, 2), 0, Fraction(-1, 2)))


def rational_phase(phase):
    """Normalize an exact coefficient of pi without quantizing its angle."""
    from .angles import Phase, phase as general_phase
    if isinstance(phase, Phase):
        return general_phase(phase)
    if isinstance(phase, float):
        raise ValueError("Use an exact integer/Fraction phase, not float")
    return Fraction(phase) % 2


def is_quarter(phase):
    """Whether a phase belongs to the finite Clifford+T phase domain."""
    from .angles import Phase
    return not isinstance(phase, Phase) and (Fraction(phase) * 4).denominator == 1


def quarter(phase):
    """Accept exact pi/4 multiples only; never round a floating angle."""
    p = rational_phase(phase)
    if not is_quarter(p):
        raise ValueError("This kernel covers Clifford+T phases (pi/4 multiples)")
    return p
