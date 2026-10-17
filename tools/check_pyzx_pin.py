"""Reject shadowed or modified PyZX dependencies before any training starts."""
import hashlib
import json
from pathlib import Path
import pyzx

root = Path(pyzx.__file__).resolve().parent
manifest = json.loads((Path(__file__).resolve().parents[1]/"docs/pyzx-pin.json").read_text())
actual = {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
          for f in sorted(root.rglob("*.py"))}
if actual != manifest["files"]:
    raise RuntimeError("Imported PyZX differs from pinned paper fork: "+str(root))
print(json.dumps(dict(status="passed", source=str(root), version=pyzx.__version__,
                     files=len(actual), commit=manifest["commit"])))
