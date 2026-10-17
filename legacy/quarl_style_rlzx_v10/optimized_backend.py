"""Exact-cost CPU optimizations for the fixed-wire total-NAM-gate objective.

No approximation of cost, CFlow settings, or reward. Unknown gate types retain
the original converter. The graph rewrite prefix is compiled from upstream.
"""
import ast
import hashlib
import inspect
import math
import pickle
import textwrap
import time
from collections import OrderedDict, defaultdict

import pyzx as zx
from qiskit import QuantumCircuit, qasm2
from cflow_exact import flow as exact_flow


def compile_rewrite_prefix(original):
    tree = ast.parse(textwrap.dedent(inspect.getsource(original.step)))
    function = tree.body[0]
    prefix = []
    for node in function.body:
        prefix.append(node)
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and
                ast.unparse(node.targets[0]) == 'self.graph' and
                ast.unparse(node.value) == 'self.graph.copy()'):
            break
    else:
        raise RuntimeError('Upstream rewrite boundary changed; refuse unreviewed adaptation')
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and
                   n.func.attr == 'extract_circuit' for node in prefix for n in ast.walk(node))
    prefix_hash = hashlib.sha256(ast.dump(ast.Module(body=prefix, type_ignores=[])).encode()).hexdigest()
    function.name = '_rewrite_prefix'
    function.body = prefix + [ast.parse('return act_type, action_id, node').body[0]]
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = dict(original.step.__globals__)
    exec(compile(module, inspect.getsourcefile(original.step), 'exec'), namespace)
    return namespace['_rewrite_prefix'], prefix_hash


def rewrite_only_class(original):
    apply, prefix_hash = compile_rewrite_prefix(original)

    class RewriteOnlyEnv(original):
        rewrite_prefix_sha256 = prefix_hash

        def reset(self, *args, **kwargs):
            self._matching_graph = self._cached_lcomp = self._cached_ids = None
            graph, info = super().reset(*args, **kwargs)
            self._last_graph_obs = info['graph_obs']
            return graph, info

        def match_lcomp(self, *args, **kwargs):
            if args or kwargs:
                return super().match_lcomp(*args, **kwargs)
            self._check_matching_graph()
            if self._cached_lcomp is None:
                self._cached_lcomp = super().match_lcomp()
            return self._cached_lcomp.copy()

        def match_ids(self, *args, **kwargs):
            if args or kwargs:
                return super().match_ids(*args, **kwargs)
            self._check_matching_graph()
            if self._cached_ids is None:
                self._cached_ids = super().match_ids()
            return self._cached_ids.copy()

        def _check_matching_graph(self):
            if getattr(self, '_matching_graph', None) is not self.graph:
                self._matching_graph = self.graph
                self._cached_lcomp = self._cached_ids = None

        def step(self, action):
            action = int(action)
            ids = self._last_graph_obs[0][3]
            if not bool((ids == action).any()) or action == -1:
                raise ValueError(f'Illegal ZX rewrite action {action}')
            if action != self.shape * (self.shape + 1) + 1:
                self._matching_graph = self._cached_lcomp = self._cached_ids = None
            act_type, action_id, nodes = apply(self, action)
            if act_type == 'STOP':
                # Upstream prefix still makes its graph copy/relabel. All input
                # states here are already relabeled by reset/previous step.
                # No topology changes: the same matches/observations are valid.
                done = True
                obs = self._last_graph_obs
                if getattr(self, '_matching_graph', None) is not None:
                    self._matching_graph = self.graph
            else:
                self.pivot_info_dict = (self.match_pivot_parallel() |
                                        self.match_pivot_boundary() | self.match_pivot_gadget())
                self.gadget_info_dict, self.gadgets = self.match_phase_gadgets()
                self.gadget_fusion_ids = list(self.gadget_info_dict)
                remaining = (len(self.match_lcomp()) + len(self.pivot_info_dict) +
                             len(self.match_ids()) + len(self.gadget_fusion_ids))
                done = remaining == 0 or self.episode_len >= self.max_episode_len
                obs = [self.policy_obs(), self.value_obs()]
                self._last_graph_obs = obs
            # The caller MUST score before committing a transition. Completion
            # exceptions propagate and cannot enter its archive/rollout/output.
            return self.graph, 0., bool(done), False, dict(
                action=action_id, nodes=nodes, graph_obs=obs, native_reward_skipped=True)

    return RewriteOnlyEnv


def fast_nam(circuit, reference_converter):
    basic = circuit.to_basic_gates()
    allowed = {zx.gates.HAD, zx.gates.CNOT, zx.gates.CZ, zx.gates.NOT,
               zx.gates.ZPhase, zx.gates.Z, zx.gates.S, zx.gates.T}
    if any(type(gate) not in allowed for gate in basic.gates):
        return reference_converter(circuit), True
    out = QuantumCircuit(basic.qubits)
    for gate in basic.gates:
        kind = type(gate)
        if kind is zx.gates.HAD:
            out.h(gate.target)
        elif kind is zx.gates.CNOT:
            out.cx(gate.control, gate.target)
        elif kind is zx.gates.CZ:
            out.h(gate.target)
            out.cx(gate.control, gate.target)
            out.h(gate.target)
        elif kind is zx.gates.NOT:
            out.x(gate.target)
        else:
            theta = float(gate.phase) * math.pi
            out.rz(theta, gate.target)
            if kind in {zx.gates.Z, zx.gates.S, zx.gates.T}:
                out.global_phase += theta / 2.
    return out, False


class ExactCompletion:
    """Same completion, with a second exact cache after first extraction/basic."""
    def __init__(self, clean, converter, counter, capacity=256, circuit_cache=True,
                 direct_conversion=True, fast_flow=True, verify_flow=False, record_requests=False):
        self.clean, self.converter, self.counter = clean, converter, counter
        self.capacity = capacity
        self.circuit_cache = circuit_cache
        self.direct_conversion = direct_conversion
        self.fast_flow, self.verify_flow = fast_flow, verify_flow
        self.flow_statistics = {}
        self.cache = OrderedDict()
        self.timings = defaultdict(float)
        self.calls = self.hits = self.fallback_calls = 0
        self.cache_requests = [] if record_requests else None

    def measured(self, name, function, *args, **kwargs):
        start = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            self.timings[name] += time.perf_counter() - start

    def evaluate(self, graph):
        self.calls += 1
        g = self.measured('clone', graph.clone)
        self.measured('normalize', g.normalize)
        extracted = self.measured('gflow_extract', zx.extract_circuit, g, up_to_perm=False)
        extracted = self.measured('first_basic', self.clean, extracted)
        start = time.perf_counter()
        # Full serialized circuit: retains exact Fraction phases, wire indices,
        # gate order, qubit count and all metadata. Conservative false misses
        # are allowed; no float rounding, graph-isomorphism or free permutations.
        key = hashlib.sha256(pickle.dumps(extracted, protocol=5)).hexdigest()
        self.timings['circuit_key'] += time.perf_counter() - start
        if self.cache_requests is not None:
            self.cache_requests.append(key)
        if self.circuit_cache and key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        g = self.measured('back_to_graph', extracted.to_graph)
        self.measured('teleport_reduce', zx.teleport_reduce, g)
        self.measured('to_graph_like', zx.to_graph_like, g)
        # Preserve the original clone boundary: PyZX graphs carry phase-tracking
        # metadata as well as adjacency, so topology ownership alone is weaker.
        g = self.measured('pre_cflow_clone', g.clone)
        if self.fast_flow:
            self.measured('cflow', exact_flow, g, verify=self.verify_flow,
                          statistics=self.flow_statistics, cFlow=True,
                          max_lc_unfusions=2, max_p_unfusions=2)
        else:
            self.measured('cflow', zx.flow_2Q_simp, g, cFlow=True,
                          max_lc_unfusions=2, max_p_unfusions=2)
        c = self.measured('cflow_extract', zx.extract_simple, g, up_to_perm=False)
        c = self.measured('second_basic', self.clean, c)
        start = time.perf_counter()
        if self.direct_conversion:
            circuit, fallback = fast_nam(c, self.converter)
            self.fallback_calls += int(fallback)
        else:
            circuit = self.converter(c)
        self.timings['nam_conversion'] += time.perf_counter() - start
        counts = self.measured('count', self.counter, circuit)
        text = self.measured('qasm_dump', qasm2.dumps, circuit) + '\n'
        result = dict(qasm=text, sha256=hashlib.sha256(text.encode()).hexdigest(), cost=counts)
        if self.circuit_cache:
            self.cache[key] = result
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        return result

    def summary(self):
        return dict(calls=self.calls, extracted_circuit_cache_hits=self.hits,
                    conversion_fallback_calls=self.fallback_calls,
                    cache_requests=self.cache_requests,
                    cflow_statistics=dict(self.flow_statistics),
                    circuit_cache_size=len(self.cache), timings=dict(self.timings))
