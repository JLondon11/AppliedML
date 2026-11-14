"""Validate experiment source identity and artifact hashes for Chapter 6.

Example:
  python verify_source_provenance.py --source vae_pca_real_digits.py \
    --expected-git-blob 666cac09a9ec5b127f2ba7d89f109c6b10065df4 \
    --outputs outputs/vae_pca --manifest evidence_manifest.json

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
    files = manifest.get("files", {})
    if not isinstance(files, dict) or not files:
        add("manifest", False, "Expected nonempty files dict of relative paths to SHA-256 hashes")
    else:
        for relative, expected in sorted(files.items()):
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                add(relative, False, "Unsafe relative path")
                continue
            candidate = outputs / relative_path
            if not candidate.is_file():
                add(relative, False, "File missing")
                continue
            actual = digest(candidate)
            add(relative, actual.lower() == str(expected).lower(),
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
