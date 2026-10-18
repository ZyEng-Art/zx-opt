"""Exact resource-feasibility masks; no cost or extraction lookahead.

Previewing a rewrite is more work than learning a zero-reward invalid self-loop,
but defines the categorical/PPO density over actual bounded legal actions.
"""
from collections import Counter, OrderedDict
import hashlib
import json


def signature(raw, action, parent_key=None):
    return hashlib.sha256(((parent_key or raw.digest())+json.dumps(action.record(), sort_keys=True)).encode()).hexdigest()


class FeasibilityMismatch(RuntimeError):
    pass


class Feasibility:
    def __init__(self, vertices, edges, capacity=8192, mode="reference"):
        if mode not in ("reference", "bounds"):
            raise ValueError("Unknown feasibility mode: "+mode)
        self.mode = mode
        self.vertices = vertices; self.edges = edges; self.capacity = capacity
        self.cache = OrderedDict(); self.stats = Counter(); self.masked_families = Counter()

    def prepare(self, raw, view, available, kernel):
        selected = []; previews = {}; parent_key = raw.digest()
        for action in available:
            key = signature(raw, action, parent_key)
            if key in self.cache:
                valid = self.cache[key]; self.cache.move_to_end(key); self.stats["cache_hits"] += 1
            else:
                bound = (kernel.resource_upper_bound(raw, view, action, parent_key)
                         if self.mode == "bounds" else None)
                if bound is not None and bound[0] <= self.vertices and bound[1] <= self.edges:
                    # Upper bounds can prove acceptance, never rejection. HH
                    # cancellation can make an oversized intermediate fit.
                    valid = True; self.stats["bound_accepts"] += 1
                    self.stats["bound_accepts_"+action.mode] += 1
                else:
                    self.stats["previews"] += 1
                    try:
                        child, auxiliary = kernel.preview(raw, view, action)
                        valid = len(child.nodes) <= self.vertices and len(child.edges) <= self.edges
                        if valid:
                            previews[key] = (child, auxiliary)
                        else:
                            self.stats["resource_masks"] += 1
                    except TimeoutError:
                        raise
                    except Exception:
                        if action.mode == "jpv":
                            raise  # A broken trusted proof is an implementation error.
                        valid = False; self.stats["unexecutable_masks"] += 1
                self.cache[key] = valid
                if len(self.cache) > self.capacity:
                    self.cache.popitem(last=False)
            if valid:
                selected.append(action)
            else:
                self.masked_families[action.family] += 1
        self.stats["calls"] += 1
        self.stats["proposed"] += len(available)
        self.stats["eligible"] += len(selected)
        return selected, previews

    def summary(self):
        return dict(self.stats, mode=self.mode, cache_size=len(self.cache), cache_capacity=self.capacity,
                    masked_families=dict(self.masked_families))
