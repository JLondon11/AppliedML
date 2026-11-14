"""Fail-closed Chapter 6 release gate; scientific scopes != manuscript approval.

Usage:
  python chapters/generative_ai_part_i/qa/check_ch6_release_gate.py --repo-root . --report release_gate.json

Exit 0 only if the current manuscript, Figure 6.20's provenance, and the
distinct scientific scope register pass structural evidence gates. Manual
scientific review is still required before final production approval.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

CHAPTER = Path("chapters/generative_ai_part_i")
EXPECTED = {f"CH6-D{i:02d}" for i in range(1, 7)}
FUNDAMENTAL = ("R1", "R3", "R4", "R8", "R9", "R10")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(root: Path) -> dict:
    chapter = root / CHAPTER
    results = []

    def record(name: str, ok: bool, detail: str) -> None:
        results.append({"gate": name, "pass": bool(ok), "detail": str(detail)})

    register = chapter / "qa/CH6_SCOPED_SCIENTIFIC_VS_PRODUCTION_REGISTER_20261113.csv"
    if not register.is_file():
        record("scoped_register", False, "Register missing")
    else:
        with register.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        ids = [r.get("unit_id", "") for r in rows]
        record("six_distinct_scoped_ids", len(rows) == 6 and set(ids) == EXPECTED,
               "IDs: " + ", ".join(ids))
        valid = all(r.get("level") == ("B" if r.get("unit_id") == "CH6-D04" else "A")
                    and r.get("scientific_status") == "SCIENTIFIC_CERTIFIED"
                    and r.get("production_status") == "PRODUCTION_HOLD"
                    and all(r.get(key) == "PASS" for key in FUNDAMENTAL)
                    and all(r.get(f"R{i}") == "PASS" for i in range(1, 11))
                    and bool(r.get("evidence_limits", "").strip())
                    for r in rows)
        record("scoped_certification_claims", valid,
               "Six scoped decisions must retain evidence limits and production HOLD")

    manifest_path = chapter / "quality_assurance/chapter6_evidence_manifest.json"
    if not manifest_path.is_file():
        record("figure_manifest", False, "No manuscript-to-evidence manifest")
    else:
        try:
            fig_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            candidates = [f for f in fig_data.get("figures", [])
                          if str(f.get("figure_id")) == "6.20"]
            if len(candidates) != 1:
                record("figure_620_mapping", False, "Figure 6.20 must have exactly one mapping")
            else:
                fig = candidates[0]
                same_domain = fig.get("manuscript_domain") == fig.get("evidence_domain")
                digest = fig.get("output_sha256", "")
                file = root / str(fig.get("output", ""))
                valid_digest = (isinstance(digest, str) and len(digest) == 64
                                and all(ch in "0123456789abcdef" for ch in digest))
                source_rel = Path(str(fig.get("source_code", "")))
                output_rel = Path(str(fig.get("output", "")))
                safe = (not source_rel.is_absolute() and ".." not in source_rel.parts
                        and not output_rel.is_absolute() and ".." not in output_rel.parts)
                source_ok = safe and (root / source_rel).is_file()
                output_ok = safe and file.is_file() and valid_digest and sha256(file) == digest
                caption_ok = bool(str(fig.get("manuscript_caption", "")).strip())
                record("figure_620_mapping",
                       fig.get("manuscript_mapping_verified") is True
                       and fig.get("experiment_executed") is True
                       and fig.get("dataset_provenance_verified") is True
                       and same_domain and source_ok and output_ok and caption_ok,
                       "Require domain-matched executable source, exact output hash, caption, and inspected manuscript mapping")
        except (ValueError, OSError, TypeError) as exc:
            record("figure_620_mapping", False, "Cannot inspect manifest: " + str(exc))

    manuscript_dir = chapter / "manuscript"
    tex = list(manuscript_dir.rglob("*.tex")) if manuscript_dir.is_dir() else []
    pdf = list(manuscript_dir.rglob("*.pdf")) if manuscript_dir.is_dir() else []
    record("current_manuscript_source", bool(tex),
           f"Current committed .tex files: {len(tex)}")
    record("current_manuscript_pdf", bool(pdf),
           f"Current committed .pdf files: {len(pdf)}")

    failed = sum(not x["pass"] for x in results)
    return {"chapter": 6, "release": "BLOCKED" if failed else "STRUCTURAL_GATE_PASS_REVIEW_REQUIRED",
            "passed": len(results) - failed, "failed": failed, "checks": results,
            "caveat": "Passing structural checks alone does not certify all 14 parent case studies."}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    report = check(args.repo_root.resolve())
    output = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output, encoding="utf-8")
    print(output)
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
