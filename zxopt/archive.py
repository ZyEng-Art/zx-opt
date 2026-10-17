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


class DeferredFrontier:
    """Bounded parents with remaining actions, used only after archive exhaustion.

    These entries are separate from the 64-state elite restart archive. Storing
    a parent preserves other branches when its selected child leaves the archive
    admission threshold. Capacity and all extra work are explicit in the protocol.
    """
    def __init__(self, capacity):
        if capacity < 0:
            raise ValueError("Negative frontier capacity")
        self.capacity = capacity
        self.entries = {}
        self.insertions = self.evictions = self.resumptions = 0

    def retain(self, entry, remaining):
        if remaining <= 0:
            self.entries.pop(entry.key, None)
            return
        if not self.capacity:
            return
        if entry.key not in self.entries:
            self.insertions += 1
        self.entries[entry.key] = entry
        if len(self.entries) > self.capacity:
            victim = max(self.entries, key=lambda k: score(self.entries[k].counts))
            del self.entries[victim]
            self.evictions += 1

    def pop(self, exhausted):
        for key in list(self.entries):
            if key in exhausted:
                del self.entries[key]
        if not self.entries:
            return None
        key = min(self.entries, key=lambda k: score(self.entries[k].counts))
        self.resumptions += 1
        return self.entries.pop(key)

    def summary(self):
        return dict(size=len(self.entries), capacity=self.capacity,
                    insertions=self.insertions, evictions=self.evictions,
                    resumptions=self.resumptions)
