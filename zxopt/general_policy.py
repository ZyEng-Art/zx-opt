"""PPO action ranking on general ZX diagrams and their separate scoring views."""
import math
import torch
from torch import nn
from torch_geometric.data import Data, Batch

from .jpv_actions import connected_ports

FEATURES = 27
CONTEXT = 8
FAMILIES = ["LC", "PV", "PVB", "PVG", "ID", "GF", "identity_insert", "phase_split",
            "spider_split", "gadget_split", "phase_transfer", "spider_nest"]
FAMILIES += [name+":"+direction for name in ("S", "I", "CP", "B", "K", "EU", "H", "SUP", "E", "C", "BW")
             for direction in ("forward", "backward")]
BASE_FAMILIES = tuple(FAMILIES)
FAMILIES += ["A:forward", "A:backward"]


def batch(observations):
    return tuple(Batch.from_data_list([obs[i] for obs in observations]) for i in range(2))


def observation(raw, view, available, context):
    """The value graph excludes action nodes; no presumed cost for unscored raw states."""
    xs, kinds, edges = [], [], []
    raw_map, view_map = {}, {}
    reached = connected_ports(raw)
    degree = {v: 0 for v in raw.nodes}
    for a, b in raw.edges:
        degree[a] += 1; degree[b] += 1
    for v, node in raw.nodes.items():
        x = [0.] * FEATURES
        x[{"Z": 0, "X": 1, "H": 2, "B": 3}[node.kind]] = 1.
        if v in raw.ports:
            x[4] = (raw.ports.index(v)+1) / max(1, len(raw.ports))
        x[5:7] = [math.sin(math.pi*float(node.phase)), math.cos(math.pi*float(node.phase))]
        x[7] = math.log1p(degree[v])/4
        x[8] = float(v in reached)
        x[23] = len(xs)/max(1, len(raw.nodes)-1)
        x[26] = raw.loops/8
        raw_map[v] = len(xs); xs.append(x); kinds.append(0)
    for a, b in raw.edges:
        edges.extend([(raw_map[a], raw_map[b]), (raw_map[b], raw_map[a])])
    if view is not None:
        from pyzx.utils import VertexType as V, EdgeType as E
        ordered = tuple(view.inputs()) + tuple(view.outputs())
        ordered_vertices = list(view.vertices())
        max_row = max([1.] + [float(view.row(w)) for w in ordered_vertices])
        for rank, v in enumerate(ordered_vertices):
            x = [0.] * FEATURES
            x[0 if view.type(v) == V.Z else 3] = 1.
            x[4] = (ordered.index(v)+1)/max(1, len(ordered)) if v in ordered else 0.
            phase = float(view.phase(v))
            x[5:7] = [math.sin(math.pi*phase), math.cos(math.pi*phase)]
            x[7] = math.log1p(view.vertex_degree(v))/4
            x[8] = 1.; x[9] = 1.  # independently normalized scoring representation
            # Extraction is representation-sensitive: give the network the
            # actual ordering/layout instead of identifying all isomorphic views.
            x[23] = rank/max(1, len(ordered_vertices)-1)
            x[24] = float(view.row(v))/max_row
            x[25] = float(view.qubit(v))/max(1, len(view.inputs()))
            view_map[v] = len(xs); xs.append(x); kinds.append(0)
        # Explicit H nodes in the observation preserve the scoring-view edge type.
        for e in view.edges():
            a, b = (view_map[v] for v in view.edge_st(e))
            if view.edge_type(e) == E.HADAMARD:
                x = [0.] * FEATURES; x[2] = x[9] = 1.
                h = len(xs); xs.append(x); kinds.append(0)
                edges.extend([(a, h), (h, a), (h, b), (b, h)])
            else:
                edges.extend([(a, b), (b, a)])
        # Corresponding named boundary ports join the two views for message passing.
        for a, b in zip(raw.ports, ordered):
            edges.extend([(raw_map[a], view_map[b]), (view_map[b], raw_map[a])])
    base_x = torch.tensor(xs, dtype=torch.float32)
    base_edge = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    v = Data(x=base_x, edge_index=base_edge, kind=torch.tensor(kinds),
             context=torch.tensor([context], dtype=torch.float32))
    y = [-1] * len(xs)
    for i, action in enumerate(available):
        x = [0.] * FEATURES; x[10] = 1.
        x[11] = action.growth/32
        x[12] = 1. if action.direction == "forward" else -1. if action.direction == "backward" else 0.
        for j, phase in enumerate(action.phases):
            x[13+2*j:15+2*j] = [math.sin(math.pi*float(phase)), math.cos(math.pi*float(phase))]
        x[19:22] = action.shape
        x[22] = float(action.mode == "jpv")
        pos = len(xs); xs.append(x); kinds.append(FAMILIES.index(action.family)+1); y.append(i)
        mapping = raw_map if action.mode == "jpv" else view_map
        for w in action.vertices:
            if w in mapping:
                edges.extend([(mapping[w], pos), (pos, mapping[w])])
    p = Data(x=torch.tensor(xs, dtype=torch.float32), kind=torch.tensor(kinds),
             edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long),
             y=torch.tensor(y, dtype=torch.long), context=v.context.clone())
    return p, v


class Encoder(nn.Module):
    def __init__(self, width=32, general_zx=False):
        super().__init__()
        self.input = nn.Linear(FEATURES, width)
        self.kind = nn.Embedding(len(FAMILIES if general_zx else BASE_FAMILIES)+1, width, padding_idx=0)
        self.layers = nn.ModuleList([nn.Linear(2*width, width) for _ in range(2)])

    def forward(self, graph):
        h = torch.tanh(self.input(graph.x) + self.kind(graph.kind))
        source, target = graph.edge_index
        counts = torch.bincount(target, minlength=len(h)).clamp_min(1).unsqueeze(-1)
        for layer in self.layers:
            messages = torch.zeros_like(h).index_add_(0, target, h[source]) / counts
            h = torch.tanh(layer(torch.cat((h, messages), dim=-1)))
        return h


class GeneralAgent(nn.Module):
    def __init__(self, width=32, general_zx=False):
        super().__init__()
        self.actor_encoder = Encoder(width, general_zx)
        self.critic_encoder = Encoder(width, general_zx)
        self.actor_head = nn.Sequential(nn.Linear(2*width+CONTEXT, width), nn.Tanh(), nn.Linear(width, 1))
        self.value_head = nn.Sequential(nn.Linear(width+CONTEXT, width), nn.Tanh(), nn.Linear(width, 1))
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=.01)
        nn.init.zeros_(self.actor_head[-1].bias)

    @property
    def actor_gnn(self):
        return nn.ModuleList([self.actor_encoder, self.actor_head])

    def pooled(self, graph, h):
        groups = graph.batch
        count = torch.bincount(groups, minlength=graph.num_graphs).clamp_min(1).unsqueeze(-1)
        return torch.zeros((graph.num_graphs, h.shape[1]), dtype=h.dtype, device=h.device).index_add_(0, groups, h)/count

    def actor(self, graph):
        h = self.actor_encoder(graph)
        pooled = self.pooled(graph, h)
        return self.actor_head(torch.cat((h, pooled[graph.batch], graph.context[graph.batch]), dim=-1))

    def get_value(self, graph):
        pooled = self.pooled(graph, self.critic_encoder(graph))
        return self.value_head(torch.cat((pooled, graph.context), dim=-1)).flatten()

    def get_action_and_value(self, observations, action=None, device="cpu", testing=False):
        p, v = observations
        logits = self.actor(p).flatten(); values = self.get_value(v)
        positions = (p.y != -1).nonzero(as_tuple=True)[0]
        groups = p.batch[positions]; graphs = p.num_graphs
        counts = torch.bincount(groups, minlength=graphs)
        if not bool((counts > 0).all()):
            raise ValueError("Cannot sample an empty proposal list")
        selected = logits[positions]
        maximum = torch.full((graphs,), -torch.inf, device=selected.device)
        maximum.scatter_reduce_(0, groups, selected.detach(), reduce="amax", include_self=True)
        shifted = selected-maximum[groups]
        mass = torch.zeros_like(maximum).scatter_add_(0, groups, shifted.exp())
        logs = shifted-mass.log()[groups]
        entropy = torch.zeros_like(maximum).scatter_add_(0, groups, -logs.exp()*logs)
        if action is None:
            action = torch.stack([torch.multinomial(logs[groups == g].exp(), 1)[0] for g in range(graphs)])
        offsets = counts.cumsum(0)-counts
        return action, logs[offsets+action], entropy, values, None, None
