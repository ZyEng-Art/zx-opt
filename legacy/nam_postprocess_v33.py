"""Exact NAM-basis commutation cleanup; cache keyed by full serialized QASM."""
import hashlib,time
from collections import OrderedDict
from qiskit import qasm2,transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import CommutativeCancellation

def to_nam(c):
    if set(c.count_ops())<={'h','rz','x','cx'}:return c
    following=transpile(c,basis_gates=['h','rz','x','cx'],optimization_level=0,seed_transpiler=203000000,num_processes=1)
    assert following.layout is None and set(following.count_ops())<={'h','rz','x','cx'}
    return following

def cost(c):return dict(gates=len(c),cx=int(c.count_ops().get('cx',0)),depth=int(c.depth()))
class Postprocessor:
    def __init__(self,capacity=256):
        self.pass_manager=PassManager([CommutativeCancellation(basis_gates=['h','rz','x','cx'])])
        self.cache=OrderedDict();self.capacity=capacity;self.calls=self.hits=0;self.seconds=0.
    def evaluate(self,candidate):
        started=time.perf_counter();self.calls+=1
        try:
            key=candidate['sha256'];assert hashlib.sha256(candidate['qasm'].encode()).hexdigest()==key
            if key in self.cache:
                self.hits+=1;self.cache.move_to_end(key);return self.cache[key],True
            c=qasm2.loads(candidate['qasm']);assert cost(c)==candidate['cost']
            following=to_nam(self.pass_manager.run(c));assert following.layout is None
            assert following.num_qubits==c.num_qubits and set(following.count_ops())<={'h','rz','x','cx'}
            counts=cost(following)
            if (counts['gates'],counts['cx'])<(candidate['cost']['gates'],candidate['cost']['cx']):
                text=qasm2.dumps(following)+'\n'
                result=dict(qasm=text,sha256=hashlib.sha256(text.encode()).hexdigest(),cost=counts)
            else:result=candidate
            self.cache[key]=result
            if len(self.cache)>self.capacity:self.cache.popitem(last=False)
            return result,False
        finally:self.seconds+=time.perf_counter()-started
    def summary(self):return dict(calls=self.calls,hits=self.hits,seconds=self.seconds)
