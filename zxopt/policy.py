"""V49 GAT/context policy with explicit embeddings for the six new rule kinds."""
import math
import sys
import torch
from torch import nn

from .legacy import LEGACY
sys.path.insert(0, str(LEGACY / "single_circuit_rlzx_v49"))
from agent import SingleCircuitAgent
from runtime import data, batch, source_classes

KINDS = ["identity_insert", "phase_split", "spider_split", "gadget_split",
         "phase_transfer", "spider_nest"]


class ExpandedAgent(SingleCircuitAgent):
    def __init__(self, width=32):
        _, Base = source_classes(LEGACY / "upstream_source_v1")
        super().__init__(Base(None, "cpu", c_hidden=width, c_hidden_v=width), width)
        self.rule_embedding = nn.Embedding(len(KINDS) + 1, 17, padding_idx=0)
        self.phase_projection = nn.Linear(2, 17, bias=False)
        nn.init.zeros_(self.rule_embedding.weight)
        nn.init.zeros_(self.phase_projection.weight)

    @property
    def actor_gnn(self):
        return nn.ModuleList(list(super().actor_gnn) + [self.rule_embedding, self.phase_projection])

    def actor(self, p):
        adjusted = p.clone()
        adjusted.x = p.x + self.rule_embedding(p.action_kind) + self.phase_projection(p.action_phase)
        return super().actor(adjusted)


def observation(base_obs, legacy_actions, extra_actions, context, attempted=None, mapping=None):
    from .dedup import action_key
    p, v = data(base_obs)
    p.action_kind = torch.zeros(len(p.x), dtype=torch.long)
    p.action_phase = torch.zeros((len(p.x), 2), dtype=torch.float32)
    actions = [(a, identifier, pos) for pos, identifier, a in legacy_actions]
    n = len(p.x)
    xs, edges, attrs, phases, kinds = [], [], [], [], []
    for i, a in enumerate(extra_actions):
        pos = n + i
        x = [0.] * 17
        x[10] = 1.
        xs.append(x)
        kinds.append(KINDS.index(a.kind) + 1)
        phases.append([math.sin(math.pi * float(a.phase)), math.cos(math.pi * float(a.phase))])
        for vertex in sorted(set(a.vertices) | set(a.subset)):
            edges += [(vertex, pos), (pos, vertex)]
            attrs += [[0., 0., 0., 0., 0., 0., 1.]] * 2
        actions.append((a, None, pos))
    if xs:
        p.x = torch.cat((p.x, torch.tensor(xs, dtype=torch.float32)))
        p.y = torch.cat((p.y, torch.arange(9100000, 9100000 + len(xs))))
        p.action_kind = torch.cat((p.action_kind, torch.tensor(kinds)))
        p.action_phase = torch.cat((p.action_phase, torch.tensor(phases, dtype=torch.float32)))
        p.edge_index = torch.cat((p.edge_index, torch.tensor(edges).t()), dim=1)
        p.edge_attr = torch.cat((p.edge_attr, torch.tensor(attrs, dtype=torch.float32)))
    p.y[p.y == 9003001] = -1
    selected = []
    for a, identifier, pos in actions:
        if attempted is not None and action_key(a, mapping) in attempted:
            p.y[pos] = -1
        else:
            selected.append((a, identifier))
    ctx = torch.tensor([context], dtype=torch.float32)
    p.context = ctx
    v.context = ctx.clone()
    return (p, v), selected
