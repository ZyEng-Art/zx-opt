import hashlib,json,math
from pathlib import Path
import numpy as np
from qiskit import qasm2
from qiskit.quantum_info import Operator
from zxopt.verify import matrix_error
out=Path('runs/angle-launch-20261017/input-semantics.json');root=Path.cwd()
results=[]
f=root/'inputs/qft_4_numeric.qasm';actual=Operator(qasm2.loads(f.read_text())).data
idx=np.arange(16);reference=np.exp(2j*np.pi*np.outer(idx,idx)/16)/4
error=matrix_error(reference,actual);assert error<2e-8,error
results.append(dict(input=f.name,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),expected='16-point unitary positive-sign DFT, fixed Qiskit wire order, global unit phase allowed',operator_error=error))
f=root/'inputs/ising_4_numeric.qasm';actual=Operator(qasm2.loads(f.read_text())).data
H=np.array([[1,1],[1,-1]])/np.sqrt(2);X=np.array([[0,1],[1,0]]);I=np.eye(2)
def tensor(matrices):
 r=np.ones((1,1))
 for m in matrices:r=np.kron(r,m)
 return r
reference=tensor([H]*4).astype(complex)
z=np.array([[1-2*((i>>q)&1) for q in range(4)] for i in range(16)])
energy=sum(z[:,u]*z[:,v] for u,v in ((0,1),(1,2),(2,3),(3,0)))
for gamma,beta in ((.37,.23),(.61,.41)):
 reference=np.diag(np.exp(-1j*gamma*energy))@reference
 reference=tensor([np.cos(beta)*I-1j*np.sin(beta)*X]*4)@reference
error=matrix_error(reference,actual);assert error<2e-8,error
results.append(dict(input=f.name,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),expected='Product of exp(-i beta sum X) exp(-i gamma sum ZZ), two layers, H tensor4 initial layer',operator_error=error))
out.write_text(json.dumps(dict(status='passed',records=results),indent=2)+'\n');print(out.read_text())
