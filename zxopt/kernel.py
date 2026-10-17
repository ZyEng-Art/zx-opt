"""Bridge frozen legacy rewrites and explicit expansion actions."""
import contextlib
import io

from .rules import Action, apply


class Kernel:
    def __init__(self, env):
        self.env = env

    def prepare(self, graph):
        e = self.env
        e.graph = graph
        e._matching_graph = e._cached_lcomp = e._cached_ids = None
        e.pivot_info_dict = e.match_pivot_parallel() | e.match_pivot_boundary() | e.match_pivot_gadget()
        e.gadget_info_dict, e.gadgets = e.match_phase_gadgets()
        e.gadget_fusion_ids = list(e.gadget_info_dict)
        obs = [e.policy_obs(), e.value_obs()]
        e._last_graph_obs = obs
        result = []
        for pos, identifier in enumerate(obs[0][3].tolist()):
            if identifier in (-1, 9003001):
                continue
            if identifier < 0:
                action = Action("GF", tuple(e.gadget_fusion_ids[-identifier - 2]))
            elif identifier > e.shape ** 2:
                action = Action("ID", (identifier - e.shape ** 2,))
            else:
                a, b = divmod(identifier, e.shape)
                action = Action("LC", (a,)) if a == b else Action(
                    ("PV", "PVB", "PVG")[e.pivot_info_dict[(a, b)][-1]], (a, b))
            result.append((pos, identifier, action))
        return obs, result

    def step(self, graph, action, identifier=None):
        if identifier is None:
            return apply(graph, action)
        self.prepare(graph.clone())
        if action.kind == "GF":
            group = next(i for i, values in enumerate(self.env.gadget_fusion_ids)
                         if set(values) == set(action.vertices))
            original = self.env.gadget_fusion_ids[group]
            # Preserve the recorded surviving root, even if set iteration in
            # a restricted query ordered this support group differently.
            self.env.gadget_info_dict[action.vertices] = self.env.gadget_info_dict[original]
            self.env.gadget_fusion_ids[group] = action.vertices
            identifier = -group-2
        self.env.episode_len = 0
        with contextlib.redirect_stdout(io.StringIO()):
            self.env.step(identifier)
        return self.env.graph

    def prepare_at(self, graph, vertex):
        """Query frozen rule predicates only at a chosen scoring-view vertex.

        The proxy limits starting vertices/edges while retaining complete host
        neighborhoods. In particular it is NOT an induced neighborhood graph:
        that would hide wires and could admit invalid rewrites.
        """
        e = self.env
        class QueryGraph:
            def __init__(self, starts):
                self.starts = starts
            def vertex_set(self):
                return set(self.starts)
            def edge_set(self):
                return set(graph.incident_edges(vertex))
            def __getattr__(self, name):
                return getattr(graph, name)

        try:
            e.graph = QueryGraph([vertex])
            e._matching_graph = e._cached_lcomp = e._cached_ids = None
            e.pivot_info_dict = e.match_pivot_parallel() | e.match_pivot_boundary() | e.match_pivot_gadget()
            lc, ids = e.match_lcomp(), e.match_ids()
            # Gadget fusion is indexed by its support. Only the group containing
            # the chosen root/leaf is passed to the frozen predicate.
            from pyzx.utils import VertexType as V
            support = None
            leaf_starts = []
            if graph.type(vertex) == V.Z:
                if graph.vertex_degree(vertex) == 1:
                    root = next(iter(graph.neighbors(vertex)))
                    support = frozenset(set(graph.neighbors(root)) - {vertex})
                else:
                    for w in graph.neighbors(vertex):
                        if graph.type(w) == V.Z and graph.vertex_degree(w) == 1 and graph.phase(w):
                            support = frozenset(set(graph.neighbors(vertex)) - {w}); break
            if support is not None:
                for w in graph.vertices():
                    if graph.type(w) == V.Z and graph.vertex_degree(w) == 1:
                        r = next(iter(graph.neighbors(w)))
                        if frozenset(set(graph.neighbors(r)) - {w}) == support:
                            leaf_starts.append(w)
            e.graph = QueryGraph(leaf_starts)
            e.gadget_info_dict, e.gadgets = e.match_phase_gadgets()
            e.gadget_info_dict = {group: value for group, value in e.gadget_info_dict.items()
                                  if vertex in group or any(e.gadgets[r] == vertex for r in group)}
            e.gadget_fusion_ids = list(e.gadget_info_dict)
            result = [(None, v*e.shape+v, Action("LC", (v,))) for v in lc]
            result += [(None, e.shape**2+v, Action("ID", (v,))) for v in ids]
            result += [(None, a*e.shape+b, Action(("PV", "PVB", "PVG")[info[-1]], (a, b)))
                       for (a, b), info in e.pivot_info_dict.items()]
            result += [(None, -i-2, Action("GF", tuple(group))) for i, group in enumerate(e.gadget_fusion_ids)]
            return result
        finally:
            e.graph = graph
            e._matching_graph = e._cached_lcomp = e._cached_ids = None
