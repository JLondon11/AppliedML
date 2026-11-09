"""Fail-closed provenance audit for canonical Chapter 9: Generative AI Part II.

Reads frozen specifications without running the generic figure generator.
Missing, mismatched, or synthetic fallback evidence is HOLD, never VERIFIED.
"""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib

EXPECTED = 11
NEEDED = ("dataset", "generator", "experiment_protocol", "metrics_path",
          "figure_path", "model_or_method", "status")


def frozen_specs(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    blobs = [node.value.value for node in tree.body if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "SPEC_BLOB" for t in node.targets)
             and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)]
    if len(blobs) != 1:
        raise ValueError("Expected one literal SPEC_BLOB")
    specs = json.loads(zlib.decompress(base64.b64decode(blobs[0], validate=True)))
    nums = [int(s[0]) for s in specs]
    if len(nums) != len(set(nums)):
        raise ValueError("Duplicate frozen figure number")
    return specs


def audit(specs, evidence, root):
    root = Path(root).resolve()
    by_id = {}
    for record in evidence.get("figures", []):
        number = int(record["figure_number"])
        if number in by_id:
            raise ValueError("Duplicate evidence figure " + str(number))
        by_id[number] = record
    frozen = {int(row[0]) for row in specs}
    if set(by_id) - frozen:
        raise ValueError("Evidence references unknown figure(s)")
    output = []
    for spec in specs:
        n, section, asset, category, original, seed, key, hash_id = spec
        n = int(n)
        record = by_id.get(n)
        reasons = []
        if record is None:
            reasons.append("missing figure-specific experiment provenance")
        else:
            for field in NEEDED:
                if not isinstance(record.get(field), str) or not record[field].strip():
                    reasons.append("missing " + field)
            if record.get("status") != "VERIFIED":
                reasons.append("not verified")
            if record.get("generator") in (
                "chapters/_shared/figure_generation_runtime.py",
                "chapters/generative_ai_part_ii/figure_generation/generate_figures.py",
            ):
                reasons.append("generic figure fallback cannot certify an experiment")
            for kind in ("metrics", "figure"):
                rel = record.get(kind + "_path")
                checksum = record.get(kind + "_sha256")
                if not isinstance(rel, str) or not isinstance(checksum, str):
                    reasons.append("missing " + kind + " file or checksum")
                    continue
                file = (root / rel).resolve()
                if not (file == root or root in file.parents):
                    reasons.append(kind + " path escapes repository")
                elif not file.is_file():
                    reasons.append(kind + " file not committed")
                elif hashlib.sha256(file.read_bytes()).hexdigest() != checksum:
                    reasons.append(kind + " SHA256 mismatch")
        output.append({"figure": n, "section": str(section), "frozen_asset": str(asset),
                       "class": str(category), "key": str(key),
                       "status": "HOLD" if reasons else "VERIFIED",
                       "reasons": reasons})
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generator", type=Path, required=True)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()
    specs = frozen_specs(args.generator)
    if len(specs) != EXPECTED:
        p.error(f"Frozen inventory expected {EXPECTED} figures; found {len(specs)}")
    rows = audit(specs, json.loads(args.evidence.read_text()), args.root)
    n = sum(r["status"] == "VERIFIED" for r in rows)
    report = {"canonical_chapter": 9, "name": "Generative AI Part II",
              "total": len(rows), "verified": n, "hold": len(rows) - n, "figures": rows}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Chapter 9: verified {n}/{len(rows)}; HOLD {len(rows)-n}")
    for row in rows:
        if row["reasons"]:
            print(f"figure {row['figure']}: {', '.join(row['reasons'])}")
    return int(args.strict and n != len(rows))


if __name__ == "__main__":
    raise SystemExit(main())
