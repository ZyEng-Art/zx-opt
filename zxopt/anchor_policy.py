"""Anchor-first policy: sample a location before constructing its action menu.

The node selector is a learned PPO policy, rather than Quarl's temperature
sampling of critic values. PPO uses the JOINT anchor/action log probability.
Empty menus are explicit zero-reward self loops, never hidden resampling.
"""
from dataclasses import dataclass
from collections import Counter
import math

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from .complete.angles import Phase
from .general_kernel import HybridAction
from .general_policy import GeneralAgent, FEATURES, observation
from .rules import Action


@dataclass(frozen=True)
class Anchor:
    space: str
    kind: str
    position: object = None

    def record(self):
        return dict(space=self.space, kind=self.kind,
                    position=list(self.position) if isinstance(self.position, tuple) else self.position)

    def query(self):
        return self.space, self.kind, self.position


def catalog(raw, view, jpv=True):
    """Cheap topology-only choices. No applicability query or rewrite preview."""
    result = []
    if jpv:
        result += [Anchor("raw", "node", v) for v, n in raw.nodes.items() if n.kind != "B"]
        result += [Anchor("raw", "edge", i) for i in range(len(raw.edges))]
        result += [Anchor("raw", "loop", i) for i in range(raw.loops)]
        result.append(Anchor("raw", "global"))  # e.g. the empty side of E
    if view is not None and not any(isinstance(n.phase, Phase) for n in raw.nodes.values()):
        from pyzx.utils import VertexType as V, EdgeType as E
        result += [Anchor("view", "node", v) for v in view.vertices() if view.type(v) == V.Z]
        result += [Anchor("view", "edge", tuple(view.edge_st(e))) for e in view.edges()
                   if view.edge_type(e) == E.HADAMARD]
    return result


def _anchor_features(raw, view, anchor, raw_map, view_map):
    mapping = raw_map if anchor.space == "raw" else view_map
    x = [0.] * FEATURES
    x[9] = float(anchor.space == "view")
    x[10] = 1.; x[11] = -1.  # anchor token; distinct from a rewrite action
    x[19:22] = [float(anchor.kind == k) for k in ("node", "edge", "global")]
    x[26] = float(anchor.kind == "loop")
    if anchor.kind == "node":
        endpoints = (anchor.position,)
    elif anchor.kind == "edge":
        endpoints = raw.edges[anchor.position] if anchor.space == "raw" else anchor.position
    else:
        endpoints = ()
    return x, [mapping[v] for v in endpoints]


def _append(graph, raw, view, anchors, selectable):
    xs, edges, ys = [], [], []
    raw_map = {v: i for i, v in enumerate(raw.nodes)}
    view_map = {} if view is None else {v: len(raw_map)+i for i, v in enumerate(view.vertices())}
    start = len(graph.x)
    for i, anchor in enumerate(anchors):
        x, ends = _anchor_features(raw, view, anchor, raw_map, view_map)
        xs.append(x); ys.append(i if selectable else -1)
        for v in ends:
            edges.extend([(v, start+i), (start+i, v)])
    if xs:
        graph.x = torch.cat((graph.x, torch.tensor(xs, dtype=graph.x.dtype)))
        graph.kind = torch.cat((graph.kind, torch.zeros(len(xs), dtype=torch.long)))
        graph.y = torch.cat((graph.y, torch.tensor(ys, dtype=torch.long)))
    if edges:
        graph.edge_index = torch.cat((graph.edge_index, torch.tensor(edges, dtype=torch.long).t()), dim=1)
    return graph


def anchor_observation(raw, view, anchors, context, prior="uniform"):
    p, _ = observation(raw, view, [], context)
    p = _append(p, raw, view, anchors, True)
    if prior != "uniform":
        if prior != "nodes":
            raise ValueError("Unknown anchor prior")
        def group(a):
            if a.space == "raw" and a.kind == "node":
                return "raw_h" if raw.nodes[a.position].kind == "H" else "raw_zx"
            if a.kind in ("global", "loop"):
                return "scalar"
            return a.space+"_"+a.kind
        groups = [group(a) for a in anchors]; counts = Counter(groups)
        weights = dict(raw_zx=.35, raw_h=.10, view_node=.45, raw_edge=.045, view_edge=.045, scalar=.01)
        # This depends only on topology categories, before applicability queries.
        # Every retained anchor has positive support; PPO evaluates the same prior.
        p.prior = torch.zeros(len(p.x), dtype=torch.float32)
        p.prior[p.y != -1] = torch.tensor([math.log(weights[g]/counts[g]) for g in groups])
    return p


def conditional_observation(raw, view, available, context, anchor):
    p, v = observation(raw, view, available, context)
    return _append(p, raw, view, [anchor], False), v


def batch(observations):
    return tuple(Batch.from_data_list([obs[i] for obs in observations]) for i in range(3))


def draw(logits, rng):
    logits = logits.detach().double().cpu().numpy()
    probabilities = np.exp(logits-logits.max()); probabilities /= probabilities.sum()
    return min(int(np.searchsorted(np.cumsum(probabilities), rng.random(), side="right")), len(logits)-1)


def empty_action():
    return HybridAction("noop", "anchor_noop", Action("anchor_noop", ()), None, None,
                        (), (0.,)*3, (0.,)*3, 0)


class AnchorAgent(GeneralAgent):
    def actor(self, graph):
        logits = super().actor(graph)
        return logits+graph.prior.unsqueeze(-1) if "prior" in graph else logits

    def distribution(self, graph, choices):
        logits = self.actor(graph).flatten()
        positions = (graph.y != -1).nonzero(as_tuple=True)[0]
        groups = graph.batch[positions]
        logs, entropy = [], []
        for i in range(graph.num_graphs):
            scores = logits[positions[groups == i]]
            if not len(scores):
                raise ValueError("Empty categorical distribution")
            logp = scores.log_softmax(0)
            logs.append(logp[choices[i]])
            entropy.append(-(logp.exp()*logp).sum())
        return torch.stack(logs), torch.stack(entropy)

    def get_action_and_value(self, observations, action=None, device="cpu", testing=False):
        a, p, v = observations
        if action is None:
            raise ValueError("Sample anchors before matching; provide local action slots explicitly")
        anchor_logs, anchor_entropy = self.distribution(a, a.selected)
        action_logs, action_entropy = self.distribution(p, action)
        return (action, anchor_logs+action_logs, anchor_entropy+action_entropy,
                self.get_value(v), anchor_logs, action_logs)
