"""Deterministic rule allocation driven only by past total-gate improvements."""


class RuleSchedule:
    def __init__(self, mode="static", native_steps=256, expanded_steps=64, record=None):
        if mode not in ("static", "staged") or native_steps < 1 or expanded_steps < 1:
            raise ValueError("Invalid rule schedule")
        self.mode = mode; self.native_steps = native_steps; self.expanded_steps = expanded_steps
        self.without_improvement = 0 if record is None else record["without_improvement"]
        if self.without_improvement < 0:
            raise ValueError("Invalid schedule counter")

    def expanded(self):
        return (self.mode == "static" or
                self.without_improvement % (self.native_steps+self.expanded_steps) >= self.native_steps)

    def observe(self, strict_gate_improvement):
        self.without_improvement = 0 if strict_gate_improvement else self.without_improvement+1

    def record(self):
        return dict(without_improvement=self.without_improvement)
