"""Validate Chapter 5 proportional scientific certification assignments.

This validates coverage and scope integrity, NOT experimental certification.
The existing R1–R10 matrix is preserved and still controls scientific HOLD.
"""
import argparse
import csv
import json
from pathlib import Path

VALID_LEVELS = {"A", "B", "C"}
EXPECTED_IDS = {f"CV5-CS{i:02d}" for i in range(1,12)}
MUST_RESEARCH = {"CV5-CS04", "CV5-CS05"}

def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))

def verify(level_file, status_file):
    assignments=read_rows(level_file)
    statuses=read_rows(status_file)
    ids=[row["unit_id"] for row in assignments]
    if len(ids)!=11 or set(ids)!=EXPECTED_IDS or len(set(ids))!=11:
        raise ValueError("Require 11 unique Chapter 5 scientific units")
    if {row["unit_id"] for row in statuses}!=EXPECTED_IDS or len(statuses)!=11:
        raise ValueError("Unmatched historical R1–R10 evidence register")
    byid={row["unit_id"]:row for row in statuses}
    summary={"A":0,"B":0,"C":0}
    certified=[]
    for row in assignments:
        id=row["unit_id"]
        level=row["assigned_level"]
        if level not in VALID_LEVELS:
            raise ValueError(f"{id}: invalid level")
        if level=="A":
            raise ValueError(f"{id}: existing named application/case study cannot be downgraded to A without separate scope-approved retitling")
        if id in MUST_RESEARCH and level!="C":
            raise ValueError(f"{id}: cross-domain/production claim requires Level C")
        if not row["allowed_scientific_scope"].strip() or not row["unresolved_empirical_gate"].strip():
            raise ValueError(f"{id}: scope/evidence blocker missing")
        summary[level]+=1
        record=byid[id]
        if all(record[f"R{j}"]=="PASS" for j in range(1,11)):
            certified.append(id)
    return {
        "chapter":5,
        "scientific_levels":summary,
        "status":"HOLD" if len(certified)!=11 else "AWAIT_INDEPENDENT_EVIDENCE_REVIEW",
        "all_ten_pass_on_historical_register":certified,
        "note":"Level reassignment does not promote FAIL/PARTIAL evidence. Certification requires independent primary experimental artifact review."
    }

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--levels",required=True,type=Path)
    ap.add_argument("--evidence",required=True,type=Path)
    ap.add_argument("--output",type=Path)
    a=ap.parse_args()
    result=verify(a.levels,a.evidence)
    payload=json.dumps(result,indent=2)+"\n"
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(payload,encoding="utf-8")
    print(payload,end="")
