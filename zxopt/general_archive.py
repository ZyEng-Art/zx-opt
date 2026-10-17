"""Separate scored and legal-but-unscored restarts; no invented latent costs."""
from collections import OrderedDict
from dataclasses import dataclass


def score(counts):
    return counts["gates"], counts["cx"]


@dataclass
class SearchState:
    diagram: object
    last_counts: dict
    cost: object
    unscored_depth: int = 0
    distance: int = 0

    @property
    def key(self):
        return self.diagram.digest()


def following_state(previous, child, result):
    if result is None:
        return SearchState(child, previous.last_counts, None, previous.unscored_depth+1, previous.distance+1), 0.
    counts = result["cost"]
    reward = (previous.last_counts["gates"] - counts["gates"])/10
    return SearchState(child, counts, counts, 0, previous.distance+1), reward


class GeneralArchive:
    def __init__(self, root, scored_capacity=64, latent_capacity=128):
        if root.cost is None or scored_capacity < 2 or latent_capacity < 0:
            raise ValueError("Need a scored root and valid capacities")
        self.root = root; self.capacity = scored_capacity; self.latent_capacity = latent_capacity
        self.scored = OrderedDict([(root.key, root)]); self.latent = OrderedDict()
        self.insertions = self.latent_insertions = self.evictions = self.latent_evictions = 0
        self.latent_restarts = 0

    def add(self, state, start_counts):
        key = state.key
        if state.cost is None:
            if not self.latent_capacity or key in self.latent:
                return False
            self.latent[key] = state; self.latent_insertions += 1
            if len(self.latent) > self.latent_capacity:
                self.latent.popitem(last=False); self.latent_evictions += 1
            return True
        self.latent.pop(key, None)
        if key in self.scored or score(state.cost) > score(start_counts):
            return False
        self.scored[key] = state; self.insertions += 1
        if len(self.scored) > self.capacity:
            best = min(self.scored, key=lambda k: score(self.scored[k].cost))
            victim = max((k for k in self.scored if k not in (self.root.key, best)),
                         key=lambda k: score(self.scored[k].cost))
            del self.scored[victim]; self.evictions += 1
        return key in self.scored

    def choose(self, rng):
        u = rng.random()
        if u < .2:
            return self.root, "root"
        if u < .6:
            lowest = min(score(s.cost) for s in self.scored.values())
            choices = [s for s in self.scored.values() if score(s.cost) == lowest]
            return choices[int(rng.integers(len(choices)))], "best"
        if u >= .8 and self.latent:
            choices = list(self.latent.values()); self.latent_restarts += 1
            return choices[int(rng.integers(len(choices)))], "latent"
        levels = sorted({s.cost["gates"] for s in self.scored.values()})
        level = levels[int(rng.integers(len(levels)))]
        choices = [s for s in self.scored.values() if s.cost["gates"] == level]
        return choices[int(rng.integers(len(choices)))], "diverse"

    def summary(self):
        return dict(scored_size=len(self.scored), scored_capacity=self.capacity,
                    latent_size=len(self.latent), latent_capacity=self.latent_capacity,
                    insertions=self.insertions, evictions=self.evictions,
                    latent_insertions=self.latent_insertions, latent_evictions=self.latent_evictions,
                    latent_restarts=self.latent_restarts,
                    best=list(min(score(s.cost) for s in self.scored.values())))
