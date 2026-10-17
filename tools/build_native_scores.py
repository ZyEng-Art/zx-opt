"""Build the optional CPU CFlow scorer into a new task-owned output directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[1]


def build(output):
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT/"zxopt/native_scores.cpp"
    include = Path(sysconfig.get_path("include"))
    if not (include/"Python.h").exists():
        raise RuntimeError("Python development headers unavailable: "+str(include))
    compiler = shlex.split(os.environ.get("CXX") or sysconfig.get_config_var("CXX") or "c++")
    if not shutil.which(compiler[0]):
        raise RuntimeError("C++ compiler unavailable: "+compiler[0])
    library = output/("zxopt_native_scores"+sysconfig.get_config_var("EXT_SUFFIX"))
    command = [*compiler, "-std=c++17", "-O3", "-fPIC", "-shared", "-I"+str(include), str(source), "-o", str(library)]
    if sys.platform == "darwin":
        command += ["-undefined", "dynamic_lookup"]
    with (output/"build.log").open("x") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    metadata = dict(library=str(library.resolve()), library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), command=command,
                    compiler=subprocess.check_output(compiler+["--version"], text=True).splitlines()[0],
                    python=sys.version, platform=sys.platform, scope="CPU-only exact integer scores; no GPU runtime or external packages")
    (output/"build.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.output.resolve())))
