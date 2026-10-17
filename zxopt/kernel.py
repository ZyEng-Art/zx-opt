"""Bridge frozen legacy rewrites and explicit expansion actions."""
import contextlib
import io
from dataclasses import dataclass

from .rules import Action, candidates, apply


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
        self.env.episode_len = 0
        with contextlib.redirect_stdout(io.StringIO()):
            self.env.step(identifier)
        return self.env.graph
