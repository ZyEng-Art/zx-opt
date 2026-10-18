"""General-diagram states, with independent normalized scoring/macro views."""
from collections import OrderedDict
import hashlib
from dataclasses import dataclass
from fractions import Fraction as F

import pyzx as zx

from .complete.adapter import from_pyzx, to_pyzx
from .kernel import Kernel
from .rules import candidates, growth, gadgets, nest_matches
from .jpv_actions import Proposals, cancel_h_pairs
from .verify import graph_matrix, matrix_error
from .complete.angles import Phase
from .legacy import environment
from runtime import graph_key
from .scalable_verify import check_graph


class OperatorMismatch(RuntimeError):
    pass


@dataclass
class View:
    graph: object
    key: str
    error: float
    normalization_error: str = None


class Views:
    def __init__(self, inputs, reference, capacity=256):
        self.inputs = inputs; self.reference = reference; self.capacity = capacity
        self.entries = OrderedDict(); self.hits = self.misses = 0
        self.max_error = 0.

    def get(self, diagram):
        key = diagram.digest()
        if key in self.entries:
            self.hits += 1; self.entries.move_to_end(key)
            return self.entries[key]
        self.misses += 1
        pg = to_pyzx(diagram, self.inputs, evaluate_expressions=True)
        error = check_graph(self.reference, pg)
        self.max_error = max(self.max_error, error)
        if error > 2e-8:
            raise OperatorMismatch("General graph changed the fixed-wire operator: " + str(error))
        try:
            zx.to_graph_like(pg)
            pg = pg.copy()  # compact IDs before the frozen matcher's integer encoding
            normalized_error = check_graph(self.reference, pg)
            self.max_error = max(self.max_error, normalized_error)
            if normalized_error > 2e-8:
                raise OperatorMismatch("Graph-like normalization changed the operator")
            result = View(pg, graph_key(pg), max(error, normalized_error))
        except OperatorMismatch:
            raise
        except Exception as exc:
            result = View(None, None, error, type(exc).__name__+": "+str(exc))
        self.entries[key] = result
        if len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
        return result


@dataclass
class HybridAction:
    mode: str
    family: str
    payload: object
    identifier: object
    parent_view_key: str
    vertices: tuple
    phases: tuple
    shape: tuple
    growth: int
    direction: str = ""

    def record(self):
        return dict(mode=self.mode, family=self.family,
                    action=self.payload.record(), identifier=self.identifier,
                    parent_view_key=self.parent_view_key)

    @classmethod
    def from_record(cls, row):
        from .complete import RuleAction
        from .rules import Action
        record = row["action"]
        if row["mode"] == "jpv":
            payload = RuleAction.from_record(record)
        else:
            payload = Action(record["kind"], tuple(record["vertices"]),
                             F(record.get("phase", "0")), tuple(record.get("subset", ())))
        return cls(row["mode"], row["family"], payload, row["identifier"], row["parent_view_key"],
                   (), (F(0),)*3, (0., 0., 0.), 0)


def rule_allowed(profile, mode, family):
    if profile not in ("full", "no-a", "macros", "native"):
        raise ValueError("Unknown rule profile: "+profile)
    if profile == "native":
        return mode in ("legacy", "noop")
    if profile == "macros":
        return mode != "jpv"
    if profile == "no-a":
        return family not in ("A:forward", "A:backward")
    return True


class GeneralKernel:
    def __init__(self, env, seed=11, jpv=True, candidate_cap=32, matching_work=2048, general_zx=False, total_candidate_cap=0, rule_profile="full", candidate_allocation="balanced"):
        rule_allowed(rule_profile, "noop", "anchor_noop")
        self.rule_profile = rule_profile
        if candidate_allocation not in ("balanced", "native-heavy"):
            raise ValueError("Unknown candidate allocation")
        self.candidate_allocation = candidate_allocation
        self.resource_view = None; self.resource_base = None
        self.native = Kernel(env)
        self.jpv = (Proposals(seed, candidate_cap, matching_work, general=general_zx,
                              include_a=rule_profile != "no-a")
                    if jpv and rule_profile in ("full", "no-a") else None)
        self.candidate_cap = candidate_cap
        self.total_candidate_cap = total_candidate_cap; self.cap_seed = seed; self.cap_rounds = {}
        from optimized_backend import compile_rewrite_prefix
        wrapper = next(c for c in type(env).__mro__ if "rewrite_prefix_sha256" in c.__dict__)
        original = wrapper.__mro__[1]
        self.preview_prefix, self.preview_prefix_sha256 = compile_rewrite_prefix(original)
        if self.preview_prefix_sha256 != env.rewrite_prefix_sha256:
            raise AssertionError("Preview prefix differs from the frozen native rewrite")
        self.prepared_key = None; self.prepared_fields = None

    def remember_matches(self, view):
        e = self.native.env
        self.prepared_key = view.key
        self.prepared_fields = (dict(e.pivot_info_dict), dict(e.gadget_info_dict),
                                dict(e.gadgets), list(e.gadget_fusion_ids))

    def prepare(self, raw, view, max_view_vertices, anchor=None, expanded=True):
        result = []
        # An evaluated scoring view must never erase the symbolic proof state.
        # Native/macro transitions remain exact rational transitions only.
        if (view.graph is not None and not any(isinstance(n.phase, Phase) for n in raw.nodes.values())
                and (anchor is None or anchor[0] == "view")):
            before = graph_key(view.graph)
            if anchor is None:
                _, native = self.native.prepare(view.graph)
            else:
                native = self.native.prepare_at(view.graph, anchor[2]) if anchor[1] == "node" else []
            if anchor is None or anchor[1] == "node":
                self.remember_matches(view)
            extra = ([] if self.rule_profile == "native" or not expanded else
                     candidates(view.graph, max_view_vertices, self.candidate_cap,
                                None if anchor is None else (anchor[1], anchor[2])))
            for _, identifier, action in native:
                result.append(HybridAction("legacy", action.kind, action, identifier, view.key,
                                           action.vertices, (F(0),)*3, (0., 0., 0.), 0))
            for action in extra:
                result.append(HybridAction("macro", action.kind, action, None, view.key,
                                           tuple(sorted(set(action.vertices)|set(action.subset))),
                                           (action.phase, F(0), F(0)), (len(action.subset)/8, 0., 0.),
                                           growth(view.graph, action)))
            if graph_key(view.graph) != before:
                raise AssertionError("Macro preparation mutated its independent scoring view")
        if expanded and self.jpv is not None and (anchor is None or anchor[0] == "raw"):
            for action in self.jpv.build(raw, None if anchor is None else (anchor[1], anchor[2])):
                rule = action.axiom
                l, r = (rule.left, rule.right) if action.direction == "forward" else (rule.right, rule.left)
                params = rule.parameters
                touched = {w for _, w in action.match.vertices}
                for _, edge, _ in action.match.wire_edges:
                    if edge >= 0:
                        touched.update(raw.edges[edge])
                result.append(HybridAction("jpv", rule.name+":"+action.direction, action, None, None,
                                           tuple(sorted(touched)),
                                           tuple(params.get(k, F(0)) for k in ("alpha", "beta", "gamma")),
                                           (params.get("left_legs", params.get("arity", 0))/8,
                                            params.get("right_legs", 0)/8, params.get("links", 0)/8),
                                           (len(r.nodes)-len(r.ports))-(len(l.nodes)-len(l.ports)),
                                           action.direction))
        if self.total_candidate_cap and len(result) > self.total_candidate_cap:
            import numpy as np
            key = raw.digest() + ("" if anchor is None else ":"+str(anchor))
            visit = self.cap_rounds.get(key, 0); self.cap_rounds[key] = visit+1
            seed = hashlib.sha256((str(self.cap_seed)+key+":"+str(visit)).encode()).digest()
            rng = np.random.default_rng(int.from_bytes(seed[:8], "big"))
            buckets = {}
            for action in result:
                buckets.setdefault(action.family, []).append(action)
            for values in buckets.values():
                rng.shuffle(values)
            families = sorted(buckets); rng.shuffle(families)
            bounded = []
            if self.candidate_allocation == "native-heavy":
                # Family-balanced within each space; half the bounded menu is
                # reserved for native rules, one quarter each for macro/JPV.
                ordered = [buckets[f][offset] for offset in range(max(map(len, buckets.values())))
                           for f in families if offset < len(buckets[f])]
                modes = ("legacy", "macro", "jpv")
                quotas = (self.total_candidate_cap//2, self.total_candidate_cap//4,
                          self.total_candidate_cap-self.total_candidate_cap//2-self.total_candidate_cap//4)
                chosen = set()
                for mode, quota in zip(modes, quotas):
                    positions = [i for i, a in enumerate(ordered) if a.mode == mode][:quota]
                    chosen.update(positions)
                for i in range(len(ordered)):
                    if len(chosen) == self.total_candidate_cap:
                        break
                    chosen.add(i)
                return [a for i, a in enumerate(ordered) if i in chosen]
            for offset in range(max(map(len, buckets.values()))):
                for family in families:
                    if offset < len(buckets[family]):
                        bounded.append(buckets[family][offset])
                        if len(bounded) == self.total_candidate_cap:
                            return bounded
            return bounded
        return result

    def resource_upper_bound(self, raw, view, action, parent_key):
        """Counts before proven HH cancellation, which only removes resources.

        This is an acceptance certificate for trusted, freshly generated
        actions, not an applicability or semantic proof. Chosen actions still
        run the ordinary checked rewrite and all operator verification.
        Unknown/native cases use the exact preview.
        """
        if action.mode == "jpv":
            a = action.payload
            if a.parent_sha256 != parent_key or a.direction not in ("forward", "backward"):
                return None
            target = a.axiom.right if a.direction == "forward" else a.axiom.left
            # Removing each temporary degree-two interface junction reduces
            # the edge count by exactly one, including all-junction circles.
            return (len(raw.nodes)-len(a.match.vertices)+len(target.nodes)-len(target.ports),
                    len(raw.edges)-len(a.match.internal_edges)+len(a.match.wire_edges)
                    +len(target.edges)-len(target.ports))
        if action.mode != "macro" or view.graph is None or action.parent_view_key != view.key:
            return None
        if self.resource_view is not view:
            try:
                base = from_pyzx(view.graph, allow_arbitrary=raw.allow_arbitrary)
                self.resource_base = len(base.nodes), len(base.edges)
            except ValueError:
                self.resource_base = None
            self.resource_view = view
        if self.resource_base is None:
            return None
        nv, ne = self.resource_base; g = view.graph; a = action.payload
        delta_power = 0
        if a.kind in ("identity_insert", "phase_split", "spider_split"):
            dn, de = 4, 4
        elif a.kind == "phase_transfer":
            dn, de = 0, 0
        elif a.kind == "gadget_split":
            targets = next((t for r, l, t in gadgets(g) if (r, l) == a.vertices), None)
            if targets is None:
                return None
            dn, de = len(targets)+3, 2*(len(targets)+1)
            delta_power = len(targets)-1
        elif a.kind == "spider_nest":
            matched, missing = nest_matches(g, a.vertices, int(a.phase))
            dn = sum(len(t)+3 for t, _ in missing)-sum(len(t)+3 for _, _, t in matched)
            de = 2*(sum(len(t)+1 for t, _ in missing)-sum(len(t)+1 for _, _, t in matched))
            delta_power = sum(len(t)-1 for t, _ in missing)-sum(len(t)-1 for _, _, t in matched)
        else:
            return None
        if delta_power and not (g.scalar.is_zero or g.scalar.floatfactor == 0):
            def power_size(p):
                return ((2, (2 if p % 2 else 4)-p) if p < 0 else
                        (p//2+2*(p % 2), p % 2))
            old_n, old_e = power_size(g.scalar.power2)
            new_n, new_e = power_size(g.scalar.power2+delta_power)
            dn += new_n-old_n; de += new_e-old_e
        return nv+dn, ne+de

    def step(self, raw, view, action):
        if not rule_allowed(self.rule_profile, action.mode, action.family):
            raise ValueError("Action excluded by rule profile: "+action.family)
        if action.mode == "noop":
            return raw.clone(), []
        if action.mode == "jpv":
            child = action.payload.apply(raw)
        else:
            if view.graph is None or graph_key(view.graph) != action.parent_view_key:
                raise ValueError("Macro action belongs to a different normalized representation")
            pg = self.native.step(view.graph, action.payload, action.identifier)
            child = from_pyzx(pg, allow_arbitrary=raw.allow_arbitrary)
        child, receipts = cancel_h_pairs(child)
        return child, receipts

    def preview(self, raw, view, action):
        if action.mode != "legacy":
            return self.step(raw, view, action)
        if view.graph is None or graph_key(view.graph) != action.parent_view_key:
            raise ValueError("Preview belongs to a different scoring view")
        if self.prepared_key != view.key:
            self.native.prepare(view.graph); self.remember_matches(view)
        e = self.native.env
        e.graph = view.graph.clone()
        e.pivot_info_dict, e.gadget_info_dict, e.gadgets, e.gadget_fusion_ids = self.prepared_fields
        e._matching_graph = e._cached_lcomp = e._cached_ids = None
        e.episode_len = 0; e.episode_stats = dict(e.episode_stats)
        self.preview_prefix(e, action.identifier)
        return cancel_h_pairs(from_pyzx(e.graph, allow_arbitrary=raw.allow_arbitrary))
