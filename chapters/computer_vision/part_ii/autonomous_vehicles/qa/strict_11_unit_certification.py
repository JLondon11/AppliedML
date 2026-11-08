#!/usr/bin/env python3
"""Fail-closed Chapter 5 certification; existence of code is NOT experiment completion.
Accepts human-reviewed real-data evidence matrix. Zero exit only if all 11
experimental units pass EVERY criterion and scientific gaps are closed.
Run: python strict_11_unit_certification.py path/to/STRICT_11_UNIT_GATE.csv
"""
import csv
import json
import sys
from pathlib import Path

CRITERIA = (
    "topic_dataset_model_alignment",
    "original_data_heldout_split",
    "named_model_executed",
    "train_validation_test_counts",
    "task_specific_metrics_and_baseline",
    "executed_output_figure_in_unit",
    "source_code_and_result_provenance",
    "published_vs_local_results_distinguished",
    "uncertainty_failure_cases_and_limitations",
    "caption_table_text_numeric_concordance",
)

def inspect(matrix):
    with Path(matrix).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = set(CRITERIA) | {"id", "unit", "certified", "blocking_scientific_gap"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Certification columns missing: " + str(required - set(reader.fieldnames or [])))
        rows = list(reader)
    if len(rows) != 11 or sorted(int(r["id"]) for r in rows) != list(range(1,12)):
        raise ValueError("The matrix must include all eleven unique Chapter 5 units")
    units = []
    for r in rows:
        passed = [k for k in CRITERIA if r[k].strip() == "evidence_verified"]
        missing = [k for k in CRITERIA if k not in passed]
        blocker = r["blocking_scientific_gap"].strip()
        certified = not missing and not blocker
        if r["certified"].strip().upper() == "YES" and not certified:
            raise ValueError("False positive certification: " + r["unit"])
        units.append({"id":int(r["id"]),"unit":r["unit"],"passed":len(passed),
                      "missing":missing,"blocking_scientific_gap":blocker,
                      "certified":certified})
    return {"certified_units":sum(u["certified"] for u in units),
            "total_units":11,"units":units,
            "explanation":"Gate checks reviewed evidence; it does not execute models or grant approval by presence of files."}

if __name__ == "__main__":
    try:
        report = inspect(sys.argv[1])
    except (OSError, ValueError, IndexError) as exc:
        print("CERTIFICATION INPUT ERROR: " + str(exc), file=sys.stderr)
        sys.exit(1)
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["certified_units"] == 11 else 2)
