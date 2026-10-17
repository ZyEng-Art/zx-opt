# SPDX-License-Identifier: Apache-2.0
# Derived at runtime from pinned PyZX's Apache-2.0 match_pivot_unfuse.
"""Hoist phase and boundary filters without changing accepted match order."""
import ast
import inspect
import pyzx.rules as rules


def build():
    tree=ast.parse(inspect.getsource(rules.match_pivot_unfuse))
    class Transform(ast.NodeTransformer):
        def visit_For(self,node):
            self.generic_visit(node)
            if isinstance(node.target,ast.Name)and node.target.id=='subset_size_v0':
                pre=ast.parse('right_candidates = [u for size in range(max_subset_v1) for u in itertools.combinations(v1n, size) if v0 not in u and (not v1b or v1b.issubset(u)) and not (pivot_phases and pivot_phases[1] is None and not u)]').body[0]
                return [pre,node]
            if isinstance(node.target,ast.Name)and node.target.id=='neighbours_to_unfuse_0':
                assert len(node.body)==3 and isinstance(node.body[-1],ast.For)
                guard=ast.parse('if pivot_phases and pivot_phases[0] is None and not neighbours_to_unfuse_0: continue').body[0]
                node.body.insert(2,guard)
                return node
            if isinstance(node.target,ast.Name)and node.target.id=='subset_size_v1':
                inner=node.body[0]
                assert isinstance(inner,ast.For)and inner.target.id=='neighbours_to_unfuse_1'
                assert len(inner.body)==4 and isinstance(inner.body[-1],ast.Expr)
                guard=ast.parse('if pivot_phases is None and not (neighbours_to_unfuse_0 or neighbours_to_unfuse_1): continue').body[0]
                return ast.For(target=inner.target,iter=ast.Name(id='right_candidates',ctx=ast.Load()),body=[guard,inner.body[-1]],orelse=[])
            return node
    tree=Transform().visit(tree);ast.fix_missing_locations(tree)
    namespace=dict(rules.__dict__);exec(compile(tree,__file__,'exec'),namespace)
    return namespace['match_pivot_unfuse']

match_pivot_unfuse=build()
