"""Optional cost-aware replacement without modifying the frozen archive."""
from . import legacy as _legacy  # Install frozen module search paths.
from buffer import CircuitBuffer, score


class SearchBuffer(CircuitBuffer):
    def __init__(self, root, capacity=64, replace_better=False):
        super().__init__(root, capacity)
        self.replace_better = replace_better
        self.replacements = 0

    def add(self, entry, trajectory_start):
        previous = self.entries.get(entry.key)
        if previous is not None and self.replace_better:
            if (score(entry.counts) < score(previous.counts)
                    and score(entry.counts) <= score(trajectory_start)):
                self.entries[entry.key] = entry
                if entry.key == self.root.key:
                    self.root = entry
                self.replacements += 1
                return True
            return False
        return super().add(entry, trajectory_start)

    def summary(self):
        return dict(super().summary(), representative_replacements=self.replacements)
