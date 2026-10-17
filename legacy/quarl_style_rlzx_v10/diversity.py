"""Bounded cost-stratified pools and descriptive exact-state diagnostics."""
import math
from collections import Counter, OrderedDict

from buffer import CircuitBuffer, primary, score


def trim_strata(entries, capacity, protected):
    """Remove the oldest unprotected state in the most crowded cost level."""
    while len(entries) > capacity:
        counts = Counter(primary(e.counts) for e in entries.values())
        eligible = [k for k in entries if k not in protected]
        victim = max(eligible, key=lambda k: counts[primary(entries[k].counts)])
        del entries[victim]


class StratifiedBuffer(CircuitBuffer):
    """Same admission/restart distribution, balanced retention by gate count."""
    def add(self, entry, trajectory_start):
        if entry.key in self.entries or score(entry.counts) > score(trajectory_start):
            return False
        self.entries[entry.key] = entry
        self.insertions += 1
        if len(self.entries) > self.capacity:
            best = min(self.entries, key=lambda k: score(self.entries[k].counts))
            trim_strata(self.entries, self.capacity, {self.root.key, best})
            self.evictions += 1
        return entry.key in self.entries


class DiverseFrontier:
    """Retain the incumbent and nearby cost levels, with a fixed root-scale band."""
    def __init__(self, root, capacity=64, objective='nam', fraction=.1):
        assert objective == 'nam' and capacity >= 2 and fraction >= 0
        self.capacity, self.objective = capacity, objective
        self.best = score(root.counts)
        self.slack = max(1, math.ceil(primary(root.counts) * fraction))
        self.entries = OrderedDict([(root.key, root)])
        self.generation = 0

    def eligible(self, counts):
        return primary(counts) <= self.best[0] + self.slack

    def offer(self, entry):
        if not self.eligible(entry.counts):
            return False
        value = score(entry.counts)
        improved = value < self.best
        if improved:
            self.best = value
            self.generation += 1
            for key in list(self.entries):
                if not self.eligible(self.entries[key].counts):
                    del self.entries[key]
        if entry.key not in self.entries:
            self.entries[entry.key] = entry
            best = min(self.entries, key=lambda k: score(self.entries[k].counts))
            trim_strata(self.entries, self.capacity, {best})
        return improved

    def choose(self, rng, masks=None):
        candidates = [e for e in self.entries.values() if masks is None or
                      masks.available(e.key, e.payload[1][0].y.tolist())]
        if not candidates:
            return None
        best = min(score(e.counts) for e in candidates)
        if rng.random() < .7:
            choices = [e for e in candidates if score(e.counts) == best]
        else:
            levels = sorted({primary(e.counts) for e in candidates})
            level = levels[int(rng.integers(len(levels)))]
            choices = [e for e in candidates if primary(e.counts) == level]
        return choices[int(rng.integers(len(choices)))]


class Diagnostics:
    """Counts use exact pickle hashes, never claim graph-isomorphism diversity."""
    def __init__(self):
        self.states = {k: set() for k in ['ft', 'pgs']}
        self.pairs = {k: set() for k in ['ft', 'pgs']}
        self.common_states, self.common_pairs = set(), set()
        self.repeated = Counter()
        self.starts = {k: Counter() for k in ['ft', 'pgs']}
        self.restart_reasons = Counter()
        self.interruptions = Counter()
        self.seconds = Counter()
        self.offers = Counter()
        self.blocked = 0

    def observe(self, lane, old_key, key, action):
        other = 'pgs' if lane == 'ft' else 'ft'
        self.repeated[lane] += int(key in self.states[lane])
        self.repeated[lane + '_pair'] += int((old_key, action) in self.pairs[lane])
        self.states[lane].update([old_key, key])
        self.common_states.update(k for k in [old_key, key] if k in self.states[other])
        self.pairs[lane].add((old_key, action))
        if (old_key, action) in self.pairs[other]:
            self.common_pairs.add((old_key, action))

    def start(self, lane, entry, reason):
        self.starts[lane][entry.key] += 1
        self.restart_reasons[lane + ':' + reason] += 1

    def summary(self):
        return dict(unique_states={k: len(v) for k, v in self.states.items()},
                    unique_pairs={k: len(v) for k, v in self.pairs.items()},
                    common_states=len(self.common_states), common_pairs=len(self.common_pairs),
                    repeats=dict(self.repeated), starts={k: sum(v.values()) for k, v in self.starts.items()},
                    unique_start_states={k: len(v) for k, v in self.starts.items()},
                    restart_reasons=dict(self.restart_reasons), interruptions=dict(self.interruptions),
                    seconds=dict(self.seconds), offers=dict(self.offers), blocked=self.blocked)
