"""Validate Chapter 5 distinct R1-R10 scientific and manuscript-production gates.

This verifies internal register consistency, not independent model execution.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

REQUIREMENTS=[f"R{i}" for i in range(1,11)]
EXPECTED={f"CV5-CS{i:02d}" for i in range(1,12)}
VALID_R={"PASS","PARTIAL","FAIL","NOT VERIFIED","N/A — JUSTIFIED"}
SCIENCE={"SCIENTIFIC_CERTIFIED","SCIENTIFIC_HOLD"}
PRODUCTION={"PRODUCTION_APPROVED","PRODUCTION_HOLD"}

def check(path):
    with Path(path).open(newline="",encoding="utf-8") as fh:
        reader=csv.DictReader(fh)
        required={"unit_id","level","unit","primary_justification","scientific_status","production_status",*REQUIREMENTS}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("incomplete scientific/production schema")
        rows=list(reader)
    ids=[r["unit_id"] for r in rows]
    if len(ids)!=11 or set(ids)!=EXPECTED or len(set(ids))!=11:
        raise ValueError("all 11 distinct Chapter 5 unit IDs required")
    stats=Counter()
    science=Counter()
    production=Counter()
    for row in rows:
        ident=row["unit_id"]
        if row["level"] not in {"B","C"}:
            raise ValueError(ident+": invalid assigned level")
        if ident in {"CV5-CS04","CV5-CS05"} and row["level"]!="C":
            raise ValueError(ident+": research-grade scope cannot silently be downgraded")
        observed=[row[k] for k in REQUIREMENTS]
        if any(x not in VALID_R for x in observed):
            raise ValueError(ident+": invalid R1-R10 status")
        if "N/A — JUSTIFIED" in observed:
            raise ValueError(ident+": explicit independently signed N/A ledger required")
        stats.update(observed)
        status=row["scientific_status"]
        if status not in SCIENCE or row["production_status"] not in PRODUCTION:
            raise ValueError(ident+": invalid certification/production status")
        all_pass=all(x=="PASS" for x in observed)
        if (status=="SCIENTIFIC_CERTIFIED")!=all_pass:
            raise ValueError(ident+": scientific certificate inconsistent with R1-R10")
        if status=="SCIENTIFIC_HOLD" and not row["primary_justification"].strip():
            raise ValueError(ident+": missing scientific hold explanation")
        science[status]+=1
        production[row["production_status"]]+=1
    return {"units":11,"criteria":110,"requirements":dict(stats),
        "scientific_certificates":dict(science),"production_approvals":dict(production),
        "chapter_scientific_complete":science["SCIENTIFIC_CERTIFIED"]==11,
        "chapter_production_complete":production["PRODUCTION_APPROVED"]==11,
        "note":"Validated status consistency, not substitute for original experimental artifact review"}

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--matrix",type=Path,required=True)
    ap.add_argument("--report",type=Path)
    args=ap.parse_args()
    report=check(args.matrix)
    payload=json.dumps(report,indent=2)+"\n"
    print(payload,end="")
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(payload,encoding="utf-8")
