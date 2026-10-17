"""Record runtime versions without creating a CUDA context."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import pyzx

versions={}
for name in ["numpy","networkx","qiskit","torch","torch-geometric","gymnasium","matplotlib"]:
    try:versions[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:versions[name]=None
root=Path(pyzx.__file__).parent
sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted(root.rglob("*.py"))}
print(json.dumps(dict(python=sys.version,platform=platform.platform(),packages=versions,
                     pyzx_version=pyzx.__version__,pyzx_source=str(root),pyzx_sources=sources),indent=2))
