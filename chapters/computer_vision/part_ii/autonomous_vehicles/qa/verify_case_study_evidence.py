#!/usr/bin/env python3
"""Audit Chapter 5 repository evidence, without inferring that models were run.

Execute from any working directory:
    python chapters/computer_vision/part_ii/autonomous_vehicles/qa/verify_case_study_evidence.py

Exit code 0: inventory structurally valid. Missing *experimental* evidence
is recorded as BLOCKED, not silently upgraded or treated as a tooling failure.
Exit code 1: broken manifest, invalid Python syntax, missing tracked script,
inconsistent approval flag, or missing case-study README.
"""
from __future__ import annotations

import ast
import csv
import json
import re
import sys
from pathlib import Path

CHAPTER = Path(__file__).resolve().parents[1]
REPO = CHAPTER.parents[3]
MANIFEST = CHAPTER / "CH5_CODE_MANIFEST.csv"
REQUIRED_COLUMNS = {
    "id", "section", "directory", "existing_related_code",
    "named_model_real_dataset_inference", "complete_executable_implementation",
    "generated_model_output_figure_verified",
    "generated_model_metrics_table_verified",
    "dataset_checkpoint_provenance_complete", "qa_status", "outstanding",
}
EVIDENCE_FLAGS = (
    "named_model_real_dataset_inference",
    "complete_executable_implementation",
    "generated_model_output_figure_verified",
    "generated_model_metrics_table_verified",
    "dataset_checkpoint_provenance_complete",
)


def audit():
    errors, warnings, cases = [], [], []
    with MANIFEST.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing_columns = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Missing manifest columns: {sorted(missing_columns)}")
        rows = list(reader)
    if len(rows) != 11:
        errors.append(f"Expected 11 case studies; found {len(rows)}")
    seen = set()
    for r in rows:
        cid = r["id"]
        if cid in seen:
            errors.append(f"Duplicate case ID {cid}")
        seen.add(cid)
        if not re.fullmatch(r"CV5-CS\d\d", cid):
            errors.append(f"Invalid case ID {cid}")
        directory = CHAPTER / r["directory"]
        readme = directory / "README.md"
        if not readme.is_file():
            errors.append(f"{cid}: missing section-specific README: {readme}")
        flags = {k: r[k].strip().upper() for k in EVIDENCE_FLAGS}
        for k, v in flags.items():
            if v not in ("YES", "NO"):
                errors.append(f"{cid}: invalid evidence flag {k}={v!r}")
        approved = r["qa_status"].strip().upper() == "APPROVED"
        if approved and not all(v == "YES" for v in flags.values()):
            errors.append(f"{cid}: APPROVED without all five evidence gates")
        scripts = [s.strip() for s in r["existing_related_code"].split(";") if s.strip()]
        present_scripts = 0
        for rel in scripts:
            item = CHAPTER / rel
            if item.suffix == ".py" and not item.is_file():
                errors.append(f"{cid}: referenced script not committed: {rel}")
            elif item.is_file() and item.suffix == ".py":
                present_scripts += 1
        local_scripts = sorted(directory.rglob("*.py")) if directory.is_dir() else []
        for s in local_scripts:
            try:
                ast.parse(s.read_text(encoding="utf-8"), filename=str(s))
            except SyntaxError as e:
                errors.append(f"{cid}: Python syntax failure: {s}: {e}")
        if not present_scripts and not local_scripts:
            warnings.append(f"{cid}: no linked executable Python implementation")
        if not approved:
            warnings.append(f"{cid}: NOT APPROVED — {r['outstanding']}")
        cases.append({
            "id": cid, "section": r["section"], "approval": r["qa_status"],
            "scripts_linked": present_scripts, "scripts_in_case_dir": len(local_scripts),
            "all_evidence_gates": all(v == "YES" for v in flags.values()),
            "outstanding": r["outstanding"],
        })
    for code in CHAPTER.rglob("*.py"):
        try:
            ast.parse(code.read_text(encoding="utf-8"), filename=str(code))
        except SyntaxError as e:
            errors.append(f"Chapter-wide syntax failure: {code.relative_to(CHAPTER)}: {e}")
    result = {
        "chapter": 5,
        "chapter_title": "Computer Vision Part II: Autonomous Systems",
        "case_count": len(rows),
        "approved_count": sum(c["approval"].upper() == "APPROVED" for c in cases),
        "unapproved_count": sum(c["approval"].upper() != "APPROVED" for c in cases),
        "cases": cases,
        "structural_errors": errors,
        "outstanding_warnings": warnings,
        "claim": "Structural source audit only; neither actual inference nor real-data metrics verified.",
    }
    return result


if __name__ == "__main__":
    report = audit()
    print(json.dumps(report, indent=2))
    sys.exit(1 if report["structural_errors"] else 0)
