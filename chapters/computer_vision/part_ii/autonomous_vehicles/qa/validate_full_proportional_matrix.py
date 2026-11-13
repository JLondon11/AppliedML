"""Validate Chapter 5 complete proportional R1–R10 status register.

This is an audit-gate checker, NOT independent proof of model inference.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

EXPECTED={f"CV5-CS{i:02d}" for i in range(1,12)}
VALID={"PASS","PARTIAL","FAIL","NOT VERIFIED","N/A — JUSTIFIED"}
FIELDS={"unit_id","level","unit","primary_justification", *(f"R{i}" for i in range(1,11))}
MANDATORY_EMPIRICAL={"R1","R3","R4","R8","R9","R10"}

def audit(path):
    with Path(path).open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        if not FIELDS.issubset(set(reader.fieldnames or [])):
            raise ValueError("Incomplete proportional matrix schema")
        rows=list(reader)
    ids=[r["unit_id"] for r in rows]
    if len(rows)!=11 or set(ids)!=EXPECTED or len(ids)!=len(set(ids)):
        raise ValueError("Expected exactly 11 unique CS01–CS11 units")
    totals=Counter()
    candidates=[]
    for r in rows:
        unit=r["unit_id"]
        if r["level"] not in {"B","C"}:raise ValueError(f"{unit}: unauthorized level downgrade")
        if unit in {"CV5-CS04","CV5-CS05"} and r["level"]!="C":
            raise ValueError(f"{unit}: research or production claim requires C")
        if not r["unit"].strip():raise ValueError(f"{unit}: missing title")
        statuses=[r[f"R{i}"] for i in range(1,11)]
        for i,status in enumerate(statuses,1):
            if status not in VALID:raise ValueError(f"{unit} R{i}: illegal state")
            if status=="N/A — JUSTIFIED" and f"R{i}" in MANDATORY_EMPIRICAL:
                raise ValueError(f"{unit} R{i}: mandatory empirical requirement cannot be N/A")
            totals[status]+=1
        if "N/A — JUSTIFIED" in statuses:
            raise ValueError(f"{unit}: N/A requires an externally accepted individual justification ledger")
        if not all(x=="PASS" for x in statuses) and not r["primary_justification"].strip():
            raise ValueError(f"{unit}: uncertified study missing reason")
        if all(x=="PASS" for x in statuses):candidates.append(unit)
    if sum(totals.values())!=110:raise ValueError("R1–R10 count mismatch")
    return {"units":len(rows),"criteria":110,
            "level_counts":dict(Counter(r["level"] for r in rows)),
            "requirement_status_counts":dict(totals),
            "status_only_candidates":candidates,
            "scientific_gate":"HOLD" if len(candidates)!=11 else "AWAIT_INDEPENDENT_SOURCE_VERIFICATION",
            "note":"Matrix truthfully summarizes judged evidence; passing matrix validation is not scientific certification."}

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--matrix",required=True,type=Path)
    ap.add_argument("--report",type=Path)
    a=ap.parse_args()
    summary=audit(a.matrix)
    payload=json.dumps(summary,indent=2,ensure_ascii=False)+"\n"
    print(payload,end="")
    if a.report:
        a.report.parent.mkdir(parents=True,exist_ok=True)
        a.report.write_text(payload,encoding="utf-8")
