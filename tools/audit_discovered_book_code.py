#!/usr/bin/env python3
"""Verify repository-wide code discovery and report incomplete publication evidence.

No external dependencies. Reads the automatically discovered file registry and
the manually curated manuscript unit registry. All paths are checked using
'git ls-files' so a local untracked file can never count as published code.

Default: diagnostic exit code 0, even if outstanding gaps are reported.
--strict: exit 1 whenever any required manuscript unit is not certified or
a referenced file is not in Git.
"""
import argparse
import collections
import json
from pathlib import Path
import subprocess
import sys

def tracked_files(root):
    cmd = ["git", "-C", str(root), "ls-files", "-z"]
    return {p.decode() for p in subprocess.check_output(cmd).split(b"\0") if p}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--strict",action="store_true")
    p.add_argument("--json",default=None,help="Write report to this JSON path")
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    tracked=tracked_files(root)
    discovered=json.loads((root/"qa/discovered_code_units_20261109.json").read_text())
    registered=json.loads((root/"book_code_units.json").read_text())
    path_errors=[]
    by_family=collections.Counter()
    discovered_code=set()
    for u in discovered["units"]:
        by_family[u["path"].split("/")[1]] += 1
        for f in u["code"]:
            discovered_code.add(f)
            if f not in tracked:
                path_errors.append({"unit":u["path"],"path":f,"issue":"not committed"})
        if u["readme"] is not None and u["readme"] not in tracked:
            path_errors.append({"unit":u["path"],"path":u["readme"],"issue":"README not committed"})
        for f in u["evidence_files"]:
            if f not in tracked:
                path_errors.append({"unit":u["path"],"path":f,"issue":"evidence not committed"})
    unit_issues=[]
    for u in registered["units"]:
        missing=[f for f in u.get("code",[])+u.get("figure_code",[])+u.get("evidence",[]) if f not in tracked]
        empty=[field for field in ("code","figure_code","evidence") if not u.get(field)]
        if missing or empty or u.get("status")!="certified":
            unit_issues.append({"unit":u["unit"],"status":u.get("status"),"uncommitted":missing,"empty_fields":empty})
    report={"discovered_code_groups":len(discovered["units"]),
            "discovered_code_files":len(discovered_code),
            "groups_by_chapter_family":dict(sorted(by_family.items())),
            "registered_manuscript_units":len(registered["units"]),
            "manuscript_units_not_fully_certified":len(unit_issues),
            "missing_tracked_paths":len(path_errors),
            "path_errors":path_errors,
            "uncertified_units":unit_issues,
            "interpretation":"A repository inventory is not a scientific certification, and does not ensure full section coverage."}
    print(json.dumps(report,indent=2))
    if args.json:
        out=root/args.json
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(report,indent=2)+"\n")
    if path_errors or (args.strict and unit_issues):
        return 1
    return 0

if __name__=="__main__":
    sys.exit(main())
