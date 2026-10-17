"""Bounded restart-state archive; deliberately separate from the PPO rollout."""
from dataclasses import dataclass


def score(counts, objective='nam'):
    if objective == 'nam':
        return (int(counts['gates']), int(counts['cx']))
    if objective == 'cx':
        return (int(counts['cx']), int(counts['gates']))
    raise ValueError(objective)


def primary(counts, objective='nam'):
    return score(counts, objective)[0]


def reward(previous, following, objective='nam'):
    return (primary(previous, objective) - primary(following, objective)) / 10.


@dataclass
class Entry:
    key: str
    counts: dict
    payload: object
    distance: int = 0


class CircuitBuffer:
    def __init__(self, root, capacity=64, objective='nam'):
        if capacity < 2:
            raise ValueError('capacity must leave room for root and another state')
        self.root = root
        self.capacity = capacity
        self.objective = objective
        self.entries = {root.key: root}
        self.insertions = self.evictions = 0

    def add(self, entry, trajectory_start):
        # Accept equal-cost states too: their topology may expose new rewrites.
        if entry.key in self.entries or score(entry.counts, self.objective) > score(trajectory_start, self.objective):
            return False
        self.entries[entry.key] = entry
        self.insertions += 1
        if len(self.entries) > self.capacity:
            # Retain root and at least one global best. Prefer lower cost;
            # equally costly newer states displace the oldest unprotected state.
            best = min(self.entries, key=lambda k: score(self.entries[k].counts, self.objective))
            eligible = [k for k in self.entries if k not in (self.root.key, best)]
            victim = max(eligible, key=lambda k: score(self.entries[k].counts, self.objective))
            del self.entries[victim]
            self.evictions += 1
        return entry.key in self.entries

    def choose(self, mode, rng):
        if mode == 'restart':
            return self.root
        best_score = min(score(e.counts, self.objective) for e in self.entries.values())
        best = [e for e in self.entries.values() if score(e.counts, self.objective) == best_score]
        if mode == 'best':
            return best[int(rng.integers(len(best)))]
        if mode != 'diverse':
            raise ValueError(mode)
        # Explicit root exploration, best exploitation, and cost-stratified
        # diversity. These probabilities are fixed before examining results.
        u = rng.random()
        if u < .2:
            return self.root
        if u < .6:
            return best[int(rng.integers(len(best)))]
        levels = sorted({primary(e.counts, self.objective) for e in self.entries.values()})
        level = levels[int(rng.integers(len(levels)))]
        choices = [e for e in self.entries.values() if primary(e.counts, self.objective) == level]
        return choices[int(rng.integers(len(choices)))]

    def summary(self):
        return dict(size=len(self.entries), capacity=self.capacity,
                    insertions=self.insertions, evictions=self.evictions,
                    objective=self.objective,
                    best=list(min(score(e.counts, self.objective) for e in self.entries.values())),
                    max_distance=max(e.distance for e in self.entries.values()))
