"""Fail-closed metadata and file integrity checks for Chapter 6 figure evidence.

Usage: python verify_figure_evidence.py chapter6_evidence_manifest.json --root REPO_ROOT
Passing does not by itself establish scientific correctness or publication quality.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def verify(manifest, root):
    issues = []
    if not isinstance(manifest, dict) or not isinstance(manifest.get("figures"), list):
        return ["Manifest must contain a figures array"]
    if not manifest["figures"]:
        return ["Manifest has no figures"]
    root = Path(root).resolve()
    seen = set()
    for row in manifest["figures"]:
        if not isinstance(row, dict):
            issues.append("Figure record is not an object")
            continue
        fid = row.get("figure_id", "<missing>")
        if not isinstance(fid, str) or not fid.strip():
            issues.append("Figure record has no usable figure_id")
            fid = "<missing>"
        if fid in seen:
            issues.append(f"{fid}: duplicate figure ID")
        seen.add(fid)
        for field in ("manuscript_domain", "evidence_domain", "dataset",
                      "source_code", "output", "output_sha256"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                issues.append(f"{fid}: missing {field}")
        # Matching figure numbers do not establish manuscript-to-output lineage.
        if row.get("manuscript_mapping_verified") is not True:
            issues.append(f"{fid}: manuscript-to-output mapping not verified")
        if row.get("manuscript_domain") != row.get("evidence_domain"):
            issues.append(f"{fid}: domain mismatch")
        expected = row.get("output_sha256")
        valid_digest = isinstance(expected, str) and bool(_SHA256.fullmatch(expected))
        if not valid_digest:
            issues.append(f"{fid}: output_sha256 must be 64 lowercase hex characters")
        for field in ("source_code", "output"):
            rel = row.get(field)
            if not isinstance(rel, str) or not rel.strip():
                continue
            path = (root / rel).resolve()
            if not path.is_relative_to(root):
                issues.append(f"{fid}: {field} escapes repository root")
            elif not path.is_file():
                issues.append(f"{fid}: missing {field}: {rel}")
            elif field == "output" and valid_digest:
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    issues.append(f"{fid}: output SHA-256 mismatch")
        if row.get("experiment_executed") is not True:
            issues.append(f"{fid}: executed experiment not attested")
        if row.get("dataset_provenance_verified") is not True:
            issues.append(f"{fid}: dataset provenance not verified")
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    issues = verify(manifest, args.root)
    for issue in issues:
        print("FAIL:", issue)
    if issues:
        print(f"BLOCKED: {len(issues)} issue(s)")
        return 1
    print("Metadata checks passed; independent scientific review still required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
