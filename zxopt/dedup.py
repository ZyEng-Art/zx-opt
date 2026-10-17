"""WL buckets followed by exact labelled isomorphism, with fixed boundary order.

Layout is intentionally ignored here. This index must NOT key an extraction-cost
cache: extraction heuristics can depend on labels and layout. Use the legacy
serialized-state cache for costs. Scalars are retained conservatively.
"""
from dataclasses import dataclass, field
from fractions import Fraction
import json

import networkx as nx


def labelled_graph(graph):
    ins = {v: i for i, v in enumerate(graph.inputs())}
    outs = {v: i for i, v in enumerate(graph.outputs())}
    result = nx.Graph()
    for v in graph.vertices():
        phase = Fraction(graph.phase(v)) % 2
        label = (int(graph.type(v)), phase.numerator, phase.denominator,
                 ins.get(v, -1), outs.get(v, -1), bool(graph.is_ground(v)))
        result.add_node(v, label=json.dumps(label, separators=(",", ":")))
    for edge in graph.edges():
        a, b = graph.edge_st(edge)
        result.add_edge(a, b, label=str(int(graph.edge_type(edge))))
    # PyZX's to_json() omits float factors within 1e-5 of one. Keep the full
    # scalar state here so that even a small amplitude change cannot merge.
    scalar = json.dumps(graph.scalar.__dict__, sort_keys=True, default=str,
                        allow_nan=False)
    result.graph["scalar"] = scalar
    return result


def bucket_key(graph):
    digest = nx.weisfeiler_lehman_graph_hash(
        graph, node_attr="label", edge_attr="label", iterations=3)
    return (graph.number_of_nodes(), graph.number_of_edges(),
            graph.graph["scalar"], digest)


@dataclass
class State:
    identifier: int
    graph: nx.Graph
    attempted: set = field(default_factory=set)


class StructuralIndex:
    """Run-local registry. WL collisions never suffice to merge two states."""
    def __init__(self, bucket=bucket_key):
        self.bucket = bucket
        self.buckets = {}
        self.states = []
        self.hits = 0
        self.comparisons = 0

    def intern(self, graph):
        current = labelled_graph(graph)
        key = self.bucket(current)
        for state in self.buckets.get(key, []):
            if current.graph["scalar"] != state.graph.graph["scalar"]:
                continue
            self.comparisons += 1
            matcher = nx.algorithms.isomorphism.GraphMatcher(
                current, state.graph,
                node_match=lambda a, b: a["label"] == b["label"],
                edge_match=lambda a, b: a["label"] == b["label"])
            if matcher.is_isomorphic():
                self.hits += 1
                return state, dict(matcher.mapping), False
        state = State(len(self.states), current)
        self.states.append(state)
        self.buckets.setdefault(key, []).append(state)
        return state, {v: v for v in current}, True

    def summary(self):
        return dict(states=len(self.states), hits=self.hits,
                    exact_isomorphism_comparisons=self.comparisons,
                    attempted_actions=sum(len(s.attempted) for s in self.states))


def action_key(action, mapping):
    return (action.kind, tuple(mapping[v] for v in action.vertices),
            tuple(sorted(mapping[v] for v in action.subset)), str(action.phase))


class SearchIdentity:
    """Choose pruning granularity independently of structural diagnostics.

    Serialized scope never shares attempted actions across graph representations.
    The key is the same conservative pickle SHA used by the completion cache;
    vertex IDs, layout, insertion order and other serialized attributes survive.
    """
    def __init__(self, scope):
        if scope not in ("structural", "serialized"):
            raise ValueError("Unknown deduplication scope: " + str(scope))
        self.scope = scope
        self.serialized_attempts = {}

    def key(self, state, serialized_key):
        return str(state.identifier) if self.scope == "structural" else serialized_key

    def actions(self, state, serialized_key, mapping):
        if self.scope == "structural":
            return state.attempted, mapping
        attempted = self.serialized_attempts.setdefault(serialized_key, set())
        return attempted, {vertex: vertex for vertex in mapping}

    def summary(self):
        return dict(scope=self.scope, serialized_registries=len(self.serialized_attempts),
                    serialized_attempted_actions=sum(map(len, self.serialized_attempts.values())))
