"""Wall-clock patience resets only on strict reductions in total gate count."""
import time


class GatePatience:
    def __init__(self, seconds, gates, now=None, record=None):
        if seconds < 0:
            raise ValueError("Patience must be nonnegative")
        self.seconds = seconds
        self.gates = gates
        self.last_improvement_unix = time.time() if now is None else now
        if record:
            if record["seconds"] != seconds:
                raise ValueError("Cannot change patience across full-state resume")
            self.gates = record["gates"]
            self.last_improvement_unix = record["last_improvement_unix"]

    def observe(self, gates, now=None):
        if gates >= self.gates:
            return False
        self.gates = gates
        self.last_improvement_unix = time.time() if now is None else now
        return True

    def remaining(self, now=None):
        if not self.seconds:
            return float("inf")
        return max(0., self.seconds-((time.time() if now is None else now)-self.last_improvement_unix))

    def record(self):
        return dict(seconds=self.seconds, gates=self.gates, last_improvement_unix=self.last_improvement_unix)
