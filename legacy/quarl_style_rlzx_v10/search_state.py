"""Bounded exact-state search frontier and flat-action visit masks."""
from collections import OrderedDict

from buffer import score


class BestFrontier:
    def __init__(self, root, capacity=64, objective='nam'):
        assert capacity > 0
        self.capacity, self.objective = capacity, objective
        self.best = score(root.counts, objective)
        self.entries = OrderedDict([(root.key, root)])
        self.generation = 0

    def offer(self, entry):
        value = score(entry.counts, self.objective)
        if value > self.best:
            return False
        improved = value < self.best
        if improved:
            self.entries.clear()
            self.best = value
            self.generation += 1
        if entry.key not in self.entries:
            self.entries[entry.key] = entry
            if len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
        return improved

    def choose(self, rng, masks=None):
        candidates = list(self.entries.values())
        if masks is not None:
            candidates = [entry for entry in candidates if masks.available(
                entry.key, entry.payload[1][0].y.tolist())]
        return None if not candidates else candidates[int(rng.integers(len(candidates)))]


class VisitMasks:
    """Soft visits clear once per state; hard failures persist until restart.

    IDs are scoped by exact graph hash. -1 is the observation's non-action
    sentinel; other negative IDs are valid gadget-fusion actions.
    """
    def __init__(self, capacity=256):
        assert capacity > 0
        self.capacity = capacity
        self.states = OrderedDict()
        self.evictions = self.restarts = self.soft_resets = 0

    def state(self, key):
        if key not in self.states:
            self.states[key] = dict(soft=set(), hard=set(), soft_enabled=True)
            if len(self.states) > self.capacity:
                self.states.popitem(last=False)
                self.evictions += 1
        self.states.move_to_end(key)
        return self.states[key]

    def available(self, key, legal):
        legal = {int(a) for a in legal if a != -1}
        return bool(legal - self.state(key)['hard'])

    def eligible(self, key, legal):
        state = self.state(key)
        allowed = [i for i, a in enumerate(legal)
                   if a not in state['hard'] and a not in state['soft']]
        reset = False
        if not allowed and any(a not in state['hard'] for a in legal):
            state['soft'].clear()
            state['soft_enabled'] = False
            self.soft_resets += 1
            reset = True
            allowed = [i for i, a in enumerate(legal) if a not in state['hard']]
        return allowed, dict(legal=len(legal), eligible=len(allowed),
                             hard=len(state['hard']), soft=len(state['soft']),
                             soft_reset=reset, soft_enabled=state['soft_enabled'])

    def record(self, key, action, hard=False):
        state = self.state(key)
        if hard:
            state['hard'].add(int(action))
            state['soft'].discard(int(action))
        elif state['soft_enabled']:
            state['soft'].add(int(action))

    def restart(self):
        self.states.clear()
        self.restarts += 1
