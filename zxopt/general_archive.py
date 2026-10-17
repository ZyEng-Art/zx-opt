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
    def __init__(self, root, scored_capacity=64, latent_capacity=128, qasm_capacity=0, root_signature=None):
        if root.cost is None or scored_capacity < 2 or latent_capacity < 0 or qasm_capacity < 0:
            raise ValueError("Need a scored root and valid capacities")
        if qasm_capacity and not root_signature:
            raise ValueError("QASM grouping requires the verified root completion signature")
        self.root = root; self.capacity = scored_capacity; self.latent_capacity = latent_capacity
        self.scored = OrderedDict([(root.key, root)]); self.latent = OrderedDict()
        self.insertions = self.latent_insertions = self.evictions = self.latent_evictions = 0
        self.latent_restarts = 0
        self.qasm_capacity = qasm_capacity
        self.signatures = {root.key: root_signature} if qasm_capacity else {}
        self.qasm_evictions = 0

    def __setstate__(self, values):
        self.__dict__.update(values)
        # Allow read-only diagnostics of our older trusted checkpoints. Search
        # resume still independently checks the exact source/configuration.
        self.__dict__.setdefault("qasm_capacity", 0)
        self.__dict__.setdefault("signatures", {})
        self.__dict__.setdefault("qasm_evictions", 0)

    def add(self, state, start_counts, signature=None):
        key = state.key
        if state.cost is None:
            if not self.latent_capacity or key in self.latent:
                return False
            self.latent[key] = state; self.latent_insertions += 1
            if len(self.latent) > self.latent_capacity:
                self.latent.popitem(last=False); self.latent_evictions += 1
            return True
        eligible = key not in self.scored and score(state.cost) <= score(start_counts)
        if eligible and self.qasm_capacity and not signature:
            raise ValueError("Scored QASM grouping requires a verified completion signature")
        self.latent.pop(key, None)
        if not eligible:
            return False
        self.scored[key] = state; self.insertions += 1
        if self.qasm_capacity:
            self.signatures[key] = signature
            group = [k for k in self.scored if self.signatures[k] == signature]
            if len(group) > self.qasm_capacity:
                # Raw representatives can enable different rewrites. Keep a
                # bounded number, favor compact diagrams, and retain the root.
                victim = max((k for k in group if k != self.root.key), key=self.resource_rank)
                self._remove(victim); self.qasm_evictions += 1
        if len(self.scored) > self.capacity:
            best = min(self.scored, key=lambda k: score(self.scored[k].cost))
            victim = max((k for k in self.scored if k not in (self.root.key, best)),
                         key=lambda k: ((score(self.scored[k].cost), self.resource_rank(k))
                                        if self.qasm_capacity else score(self.scored[k].cost)))
            self._remove(victim)
        return key in self.scored

    def resource_rank(self, key):
        diagram = self.scored[key].diagram
        return len(diagram.nodes), len(diagram.edges), key

    def _remove(self, key):
        del self.scored[key]
        self.signatures.pop(key, None)
        self.evictions += 1

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
        result = dict(scored_size=len(self.scored), scored_capacity=self.capacity,
                    latent_size=len(self.latent), latent_capacity=self.latent_capacity,
                    insertions=self.insertions, evictions=self.evictions,
                    latent_insertions=self.latent_insertions, latent_evictions=self.latent_evictions,
                    latent_restarts=self.latent_restarts,
                    best=list(min(score(s.cost) for s in self.scored.values())))
        if self.qasm_capacity:
            result.update(qasm_capacity=self.qasm_capacity, qasm_groups=len(set(self.signatures.values())),
                          qasm_evictions=self.qasm_evictions)
        return result
