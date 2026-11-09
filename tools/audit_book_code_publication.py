#!/usr/bin/env python3
"""Book-wide source-code publication gate.

Runs against a checked-out Git repository. Does not claim scientific correctness:
it verifies registered units have committed source, references, and evidence paths.
Usage: python tools/audit_book_code_publication.py --manifest book_code_units.json
       python tools/audit_book_code_publication.py --manifest book_code_units.json --strict
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REQUIRED=("chapter","unit","kind","code","figure_code","evidence","dataset","status")
VALID_STATUSES={"pending","blocked","executed","validated","certified"}

def git_files(root):
    process=subprocess.run(
        ["git","-C",str(root),"ls-files","-z"],
        capture_output=True,check=True)
    return {p.decode("utf-8") for p in process.stdout.split(b"\0") if p}

def audit(root,manifest):
    registered=manifest.get("units",[])
    tracked=git_files(root)
    findings=[]
    for rec in registered:
        unit=str(rec.get("unit","<unnamed>"))
        problems=[f"Missing field {field}" for field in REQUIRED if field not in rec]
        for field in ("code","figure_code","evidence"):
            entry=rec.get(field)
            if not isinstance(entry,list):
                problems.append(f"{field}: expected list of committed paths")
                continue
            if not entry:
                problems.append(f"{field}: empty path list")
            for path in entry:
                if not isinstance(path,str) or not path.startswith("chapters/"):
                    problems.append(f"{field}: invalid chapter path {path!r}")
                elif path not in tracked:
                    problems.append(f"{field}: NOT COMMITTED {path}")
        state=rec.get("status")
        if state not in VALID_STATUSES:
            problems.append("Unrecognized certification status")
        if state=="certified" and problems:
            problems.append("FALSE CERTIFICATION: required evidence missing")
        findings.append({"unit":unit,"chapter":rec.get("chapter"),
                         "status":state,"issues":problems,
                         "committed_paths_verified":not problems})
    return findings

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",default="book_code_units.json")
    parser.add_argument("--strict",action="store_true",
                        help="Fail when any registered unit is not certified")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    p=root/args.manifest
    if not p.is_file():
        print(f"Missing manifest: {p}",file=sys.stderr);return 2
    records=audit(root,json.loads(p.read_text(encoding="utf-8")))
    failed=[r for r in records if r["issues"]]
    uncertified=[r for r in records if r["status"]!="certified"]
    print(json.dumps({
        "registered_units":len(records),
        "missing_or_invalid_git_evidence":len(failed),
        "not_certified":len(uncertified),
        "records":records},indent=2))
    if failed or (args.strict and uncertified):
        return 1
    return 0

if __name__=="__main__":
    raise SystemExit(main())
