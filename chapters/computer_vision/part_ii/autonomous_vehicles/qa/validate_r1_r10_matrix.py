"""Fail-closed validation of Chapter 5 R1–R10 evidence status.

A status matrix is an audit register, not sufficient experimental proof.
Never change FAIL/PARTIAL to PASS without primary source/model/figure evidence
and an independently reviewed corresponding experiment report.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

REQUIREMENTS = tuple(f"R{i}" for i in range(1, 11))
STATUSES = {"PASS", "PARTIAL", "FAIL", "NOT VERIFIED"}
EXPECTED_UNITS = tuple(f"CV5-CS{i:02d}" for i in range(1, 12))


def evaluate(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        mandatory = {"unit_id", "unit", "primary_blocker", *REQUIREMENTS}
        if not mandatory.issubset(set(reader.fieldnames or ())):
            raise ValueError("Incomplete R1–R10 matrix schema")
        rows = list(reader)
    if len(rows) != len(EXPECTED_UNITS):
        raise ValueError("Chapter 5 registry must contain exactly 11 units")
    ids = [row["unit_id"] for row in rows]
    if tuple(sorted(ids)) != EXPECTED_UNITS:
        raise ValueError("Missing, duplicate or unexpected Chapter 5 unit ID")
    counts = Counter()
    certified = []
    for row in rows:
        if not row["unit"].strip():
            raise ValueError("Unnamed unit")
        for requirement in REQUIREMENTS:
            state = row[requirement]
            if state not in STATUSES:
                raise ValueError(f"{row['unit_id']} {requirement}: invalid state {state!r}")
            counts[state] += 1
        fully_passed = all(row[k] == "PASS" for k in REQUIREMENTS)
        if fully_passed:
            certified.append(row["unit_id"])
        elif not row["primary_blocker"].strip():
            raise ValueError(f"Uncertified unit lacks written limitation: {row['unit_id']}")
    return {
        "unit_count": len(rows),
        "requirement_count": len(rows) * len(REQUIREMENTS),
        "certified_unit_ids_by_status_only": certified,
        "full_certification_is_not_proven_by_status_matrix": True,
        "blocked_unit_ids": [row["unit_id"] for row in rows if row["unit_id"] not in certified],
        "status_counts": {k: counts[k] for k in ("PASS", "PARTIAL", "FAIL", "NOT VERIFIED")},
        "gate": "HOLD" if len(certified) != len(EXPECTED_UNITS) else "AWAIT_INDEPENDENT_EVIDENCE_AUDIT",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-complete", action="store_true",
                        help="Exit 2 if any unit lacks all ten PASS states")
    args = parser.parse_args()
    result = evaluate(args.matrix)
    message = json.dumps(result, indent=2) + "\n"
    print(message, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(message, encoding="utf-8")
    if args.require_complete and result["gate"] != "AWAIT_INDEPENDENT_EVIDENCE_AUDIT":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
