"""Validate experiment source identity and artifact hashes for Chapter 6.

Example:
  python verify_source_provenance.py --source vae_pca_real_digits.py \
    --expected-git-blob 666cac09a9ec5b127f2ba7d89f109c6b10065df4 \
    --outputs EVIDENCE_PACKAGE_ROOT --manifest artifact_manifest.json

Exits nonzero on any missing/mismatching file. Source-identity checks are NOT
scientific certification or evidence of manuscript figure integration.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\x00" + data).hexdigest()


def inspect(source: Path, expected_blob: str, outputs: Path, manifest: dict) -> dict:
    checks = []
    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "pass": bool(passed), "detail": detail})
    if not source.is_file():
        add("source", False, f"Missing: {source}")
    else:
        actual = git_blob_sha(source)
        add("git_blob_source_identity", actual.lower() == expected_blob.lower(),
            f"expected {expected_blob}; actual {actual}")
    # Accept the original generic format and the exact-head Chapter 6 package.
    # The latter includes the source script under its package-relative filename.
    has_legacy = "files" in manifest
    has_package = "files_sha256" in manifest
    if has_legacy and has_package:
        add("manifest_schema", False, "Ambiguous manifest: both files and files_sha256")
        files = {}
    elif has_package:
        files = manifest["files_sha256"]
        declared_blob = manifest.get("source_git_blob_sha1")
        add("manifest_source_blob", isinstance(declared_blob, str)
            and declared_blob.lower() == expected_blob.lower(),
            f"expected {expected_blob}; manifest {declared_blob}")
    else:
        files = manifest.get("files", {})
    if not isinstance(files, dict) or not files:
        add("manifest", False, "Expected nonempty files or files_sha256 dictionary")
    else:
        source_name = source.name
        if has_package and source_name not in files:
            add("manifest_source_sha256", False, f"Missing source entry {source_name}")
        for relative, expected in sorted(files.items()):
            rel = Path(relative)
            if rel.is_absolute() or ".." in rel.parts or not rel.parts:
                add(relative, False, "Unsafe relative path")
                continue
            # In the exact-head package the root-level source and output/ files
            # are peers; --outputs must point at their common package directory.
            candidate = source if has_package and relative == source_name else outputs / rel
            if not candidate.is_file():
                add(relative, False, f"File missing: {candidate}")
                continue
            actual = digest(candidate)
            valid = isinstance(expected, str) and len(expected) == 64 and all(
                ch in "0123456789abcdef" for ch in expected)
            add(relative, valid and actual == expected,
                f"expected {expected}; actual {actual}")
    return {"passed": sum(int(c["pass"]) for c in checks),
            "failed": sum(int(not c["pass"]) for c in checks),
            "scope": "source blob and artifact integrity only",
            "checks": checks}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--expected-git-blob", required=True)
    p.add_argument("--outputs", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    result = inspect(args.source, args.expected_git_blob, args.outputs,
                     json.loads(args.manifest.read_text(encoding="utf-8")))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
