"""Reproduce general-ZX unit, independent high-precision A, and unitary proofs."""
import argparse
import datetime
from fractions import Fraction as F
import hashlib
from itertools import product
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tests")]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def mp_tensor(diagram, dps=70, symbolic=False):
    """Independent incidence-factor elimination, without PyZX or z_network."""
    import mpmath as m
    import sympy as s
    from zxopt.complete.angles import numeric, expr
    with m.workdps(dps):
        one = s.Integer(1) if symbolic else m.mpc(1)
        root2 = s.sqrt(2) if symbolic else m.sqrt(2)
        incidence = {v: [] for v in diagram.nodes}
        for i, edge in enumerate(diagram.edges):
            for v in edge: incidence[v].append(i)
        factors = []; cache = {}
        for v, n in diagram.nodes.items():
            labels = incidence[v]
            scope = tuple(sorted(set(labels)))
            if n.kind == "B":
                boundary = len(diagram.edges)+diagram.ports.index(v)
                scope = tuple(sorted((*scope, boundary)))
            else:
                if n.phase not in cache:
                    if symbolic:
                        cache[n.phase] = s.exp(s.I*s.pi*expr(n.phase))
                    else:
                        p = numeric(n.phase, digits=dps+20)
                        cache[n.phase] = m.exp(1j*m.pi*m.mpf(p.numerator)/p.denominator)
                exponential = cache[n.phase]
            table = {}
            for values in product((0,1), repeat=len(scope)):
                bits = dict(zip(scope,values))
                raw = [bits[e] for e in labels]
                if n.kind == "B":
                    value = one*int(raw[0] == bits[boundary])
                elif n.kind == "H":
                    value = (-1 if raw[0] and raw[1] else 1)/root2
                elif n.kind == "Z":
                    value = int(all(x==0 for x in raw))+exponential*int(all(x==1 for x in raw))
                else:
                    value = (1+exponential*(-1)**sum(raw))*root2**(-len(raw))
                table[values] = value
            factors.append((scope,table))
        internal = set(range(len(diagram.edges)))
        while internal:
            def union(v): return tuple(sorted({w for scope,_ in factors if v in scope for w in scope}))
            v = min(internal,key=lambda w:(len(union(w)),w)); scope=union(v)
            if len(scope)>20: raise MemoryError("Independent verification contraction width exceeded")
            joined=[f for f in factors if v in f[0]]; factors=[f for f in factors if v not in f[0]]
            remaining=tuple(w for w in scope if w!=v); table={}
            for values in product((0,1),repeat=len(remaining)):
                bits=dict(zip(remaining,values)); total=one*0
                for bit in (0,1):
                    bits[v]=bit; value=one
                    for labels,t in joined: value*=t[tuple(bits[w] for w in labels)]
                    total+=value
                table[values]=total
            factors.append((remaining,table)); internal.remove(v)
        outputs=[]
        for values in product((0,1),repeat=len(diagram.ports)):
            bits={len(diagram.edges)+i:b for i,b in enumerate(values)}; value=one*2**diagram.loops
            for labels,table in factors: value*=table[tuple(bits[w] for w in labels)]
            outputs.append(value)
        return outputs


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("output",type=Path)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    import mpmath as m
    import sympy
    import numpy as np
    from zxopt.complete import GENERAL_NAMES, RuleAction, axiom, actions, symbol
    from zxopt.complete.angles import expr
    from zxopt.complete.adapter import to_pyzx
    from zxopt.jpv_actions import cancel_h_pairs, replay_h_pairs
    from zxopt.general_kernel import Views
    from test_general_zx import unitary_a_host

    paths=(list((ROOT/"zxopt").rglob("*.py"))+list((ROOT/"tests").glob("test_*.py"))
           +[Path(__file__).resolve(),ROOT/"requirements.txt"])
    sources={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(paths)}
    commit=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(ROOT),"status","--porcelain","--untracked-files=no"],text=True).strip()
    dump(out/"protocol.json",dict(git_commit=commit,tracked_dirty=bool(dirty),sources=sources,
        source="https://arxiv.org/pdf/1903.06035v3",basis="Figures 1/2/3 and Theorem 1.3",
        sympy=sympy.__version__,mpmath=m.__version__,resource="local CPU only",
        scope="Checked instances of general JPV schemata; finite CAS/candidate coverage is not complete search"))
    suite=unittest.defaultTestLoader.discover(str(ROOT/"tests"))
    t=time.perf_counter()
    with (out/"unit-tests.txt").open("w") as stream:
        units=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    unit_record=dict(passed=units.wasSuccessful(),count=units.testsRun,seconds=time.perf_counter()-t,
                     failures=units.failures,errors=units.errors)
    dump(out/"unit-tests.json",unit_record)
    if not units.wasSuccessful(): raise AssertionError("Unit checks failed; inspect preserved logs")
    print(json.dumps(dict(event="unit-tests",count=units.testsRun,status="passed")),flush=True)
    # Prove the literal transcription's open tensor formulas with arbitrary
    # real symbols, independently of the nonlinear solver and PyZX adapter.
    symbols={k:symbol(k) for k in ("alpha","beta","theta1","theta2","theta3","gamma")}
    a,b,t1,t2,t3,g=(expr(symbols[k]) for k in ("alpha","beta","theta1","theta2","theta3","gamma"))
    closed_left=[1+sympy.I,sympy.exp(sympy.I*sympy.pi/4)*
                 (sympy.exp(sympy.I*sympy.pi*t1)*sympy.cos(sympy.pi*a)+
                  sympy.exp(sympy.I*sympy.pi*t2)*sympy.cos(sympy.pi*b))]
    closed_right=[1+sympy.I,2*sympy.exp(sympy.I*sympy.pi*(t3+sympy.Rational(1,4)))*sympy.cos(sympy.pi*g)]
    symbolic_checks=[]
    for color in ("Z","X"):
        left=axiom("A",domain="general",color=color,
                   **{k:symbols[k] for k in ("alpha","beta","theta1","theta2")}).left
        right=axiom("A",domain="general",color=color,alpha=symbols["gamma"],beta=symbols["gamma"],
                    theta1=symbols["theta3"],theta2=symbols["theta3"],
                    theta3=symbols["theta3"],gamma=symbols["gamma"]).right
        for side,diagram,closed in (("left",left,closed_left),("right",right,closed_right)):
            expected=closed if color=="Z" else [(closed[0]+closed[1])/sympy.sqrt(2),
                                               (closed[0]-closed[1])/sympy.sqrt(2)]
            residuals=[sympy.simplify(sympy.expand_complex(x-y))
                       for x,y in zip(mp_tensor(diagram,symbolic=True),expected)]
            assert residuals==[0,0],(color,side,residuals)
            symbolic_checks.append(dict(color=color,side=side,residuals=list(map(str,residuals)),
                                         formula=list(map(str,expected)),status="proved-by-symbolic-contraction"))
    dump(out/"symbolic-a-proof.json",symbolic_checks)
    rng=random.Random(20261017);cases=[]
    samples=[(F(0),F(1,3),F(0),F(0)),(F(1,2),F(1,2),F(0),F(0)),
             (F(0),F(0),F(0),F(1)),(F(0),F(0),F(0),F(0))]
    samples += [tuple(F(rng.randint(-16,16),rng.choice((3,5,7,11))) for _ in range(4)) for _ in range(28)]
    maximum=m.mpf(0)
    with m.workdps(70):
        for color in ("Z","X"):
            for a,b,t1,t2 in samples:
                rule=axiom("A",domain="general",color=color,alpha=a,beta=b,theta1=t1,theta2=t2)
                left,right=mp_tensor(rule.left),mp_tensor(rule.right)
                error=max(abs(x-y) for x,y in zip(left,right));maximum=max(maximum,error)
                assert error<m.mpf("1e-55"),(rule.record(),str(error))
                for direction,source in (("forward",rule.left),("backward",rule.right)):
                    action=next(actions(source,rule,direction))
                    child=RuleAction.from_record(json.loads(json.dumps(action.record()))).apply(source)
                    actual=mp_tensor(child)
                    assert max(abs(x-y) for x,y in zip(actual,left))<m.mpf("1e-55")
                    cases.append(dict(rule=rule.record(),direction=direction,operator_error=str(error),
                                      action=action.record(),source=source.record(),child=child.record()))
    dump(out/"a-cases.json",cases)
    host=unitary_a_host(F(1,3))
    reference=np.diag([1.,np.exp(1j*np.pi/3)])
    views=Views(1,reference)
    root=host;steps=[]
    # Both directions inside a true unitary graph, with the symbolic result
    # retained and its saved action reconstructed before each transition.
    inverse=axiom("A",domain="general",alpha=F(0),beta=F(0),
                   theta1=F(1,3)+F(1,4),theta2=F(1,3)-F(1,4),
                   theta3=F(1,3),gamma=F(1,4))
    chosen=next(actions(host,inverse,"backward"))
    child=RuleAction.from_record(chosen.record()).apply(host)
    steps.append(dict(action=chosen.record(),source=host.record(),child=child.record(),
                      operator_error=views.get(child).error))
    forward=axiom("A",domain="general",alpha=F(0),beta=F(0),
                   theta1=F(1,3)+F(1,4),theta2=F(1,3)-F(1,4))
    chosen=next(actions(child,forward))
    host=RuleAction.from_record(chosen.record()).apply(child)
    digest=host.digest();views.get(host);assert host.digest()==digest
    steps.append(dict(action=chosen.record(),source=child.record(),child=host.record(),
                      operator_error=views.get(host).error))
    dump(out/"unitary-proof.json",dict(root=root.record(),steps=steps,status="passed"))
    for name,digest in sources.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    report=dict(status="passed",unit_tests=unit_record,axiom_families=list(GENERAL_NAMES),
                high_precision_a_cases=len(samples)*2,replayed_a_actions=len(cases),decimal_precision=70,
                max_a_tensor_error=str(maximum),unitary_steps=len(steps),max_unitary_operator_error=views.max_error,
                symbolic_transcription_proofs=len(symbolic_checks),
                elapsed_seconds=time.perf_counter()-start,source_unchanged=True,
                boundary="Published rule completeness is distinct from supported real-expression/constraint certificates and finite RL search")
    dump(out/"verification.json",report);print(json.dumps(report),flush=True)


if __name__=="__main__":main()
