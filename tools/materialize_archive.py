"""Materialize a checksum-verified final archive into a new local directory.

This verifies transport and paths. Experiment validity remains the separate
responsibility of finished.json, replay audits and result-specific checks.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def materialize(metadata_path, output):
    metadata = json.loads(metadata_path.read_text())
    archive = metadata_path.parent/PurePosixPath(metadata["path"]).name
    if archive.stat().st_size != metadata["bytes"]:
        raise ValueError("Archive size differs from its final receipt")
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != metadata["sha256"]:
        raise ValueError("Archive checksum differs from its final receipt")
    output.mkdir(parents=True, exist_ok=False)
    members = 0; uncompressed = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or not (member.isfile() or member.isdir()):
                raise ValueError("Unsupported archive member: "+member.name)
            # Output is exclusively new, and links/special files are excluded.
            # Streaming extraction avoids retaining thousands of tar headers.
            tar.extract(member, path=output)
            members += 1; uncompressed += member.size
    result = dict(status="transport_verified_and_materialized", archive=str(archive.resolve()),
                  metadata_sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                  archive_sha256=metadata["sha256"], bytes=metadata["bytes"],
                  uncompressed_bytes=uncompressed, members=members,
                  source_commit=metadata.get("source_commit"),
                  recorded_study_status=metadata.get("study_status", "legacy_see_finished_and_audit"),
                  scope="Transport verification only; this receipt is not a claim that the study passed")
    (output/"materialization.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("metadata", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    print(json.dumps(materialize(a.metadata.resolve(), a.output.resolve())))
