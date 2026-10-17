"""Opt-in exact CFlow callback; frozen legacy source files remain untouched."""
import hashlib
import importlib.util
import json
from pathlib import Path


def enable_native_flow(library):
    from .legacy import LEGACY
    import cflow_exact
    library = Path(library).resolve()
    source = Path(__file__).with_name("native_scores.cpp")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    library_sha = hashlib.sha256(library.read_bytes()).hexdigest()
    build = json.loads((library.parent/"build.json").read_text())
    if (build["source_sha256"] != source_sha or build["library_sha256"] != library_sha
            or Path(build["library"]).name != library.name):
        raise ValueError("Native CFlow binary does not match its current source/build receipt")
    spec = importlib.util.spec_from_file_location("zxopt_native_scores", library)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the requested native CFlow module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cflow_exact.native_score_batch = module.score_batch
    return dict(library=str(library), library_sha256=library_sha, source_sha256=source_sha,
                build_receipt_sha256=hashlib.sha256((library.parent/"build.json").read_bytes()).hexdigest(),
                legacy_hook_sha256=hashlib.sha256((LEGACY/"quarl_style_rlzx_v10/cflow_exact.py").read_bytes()).hexdigest(),
                scope="Exact score callback only; same match enumeration, driver, ordering and tie handling; no persistent graph cache")
