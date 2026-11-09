#!/usr/bin/env python3
"""Validate the committed real-data case-study figure coverage inventory.

IMPORTANT: Path checks are not scientific certifications.
Run: python tools/verify_real_data_figure_evidence.py
Option: --strict fails if any registered code unit lacks visible result files
        or figure assets in its own directory.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--strict",action="store_true")
    parser.add_argument("--report",help="Save diagnostic JSON to this file")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    inventory=json.loads((root/"qa/REAL_DATA_CASE_STUDY_FIGURE_EVIDENCE_AUDIT_20261109.json").read_text())
    tracked={b.decode() for b in subprocess.check_output(
        ["git","-C",str(root),"ls-files","-z"]).split(b"\0") if b}
    violations=[]
    for item in inventory["units"]:
        paths=(item["committed_code"]
               +item["visible_result_files"]
               +item["visible_result_figure_assets"]
               +item["visible_model_artifacts"])
        if item["committed_readme"]:
            paths.append(item["committed_readme"])
        missing=[path for path in paths if path not in tracked]
        no_results=not item["visible_result_files"]
        no_figures=not item["visible_result_figure_assets"]
        violations.append({"unit":item["unit"],"untracked_paths":missing,
                           "no_visible_numeric_evidence":no_results,
                           "no_visible_result_figures":no_figures,
                           "certified":False})
    findings={"registered_code_units":len(violations),
              "untracked_asset_paths":sum(len(x["untracked_paths"]) for x in violations),
              "no_visible_metrics":sum(x["no_visible_numeric_evidence"] for x in violations),
              "no_visible_figures":sum(x["no_visible_result_figures"] for x in violations),
              "scientific_certification":"NOT_PERFORMED_BY_THIS_SCRIPT",
              "note":"No figure or model claim is certified without dataset, execution, numerical and manuscript inspection.",
              "units":violations}
    output=json.dumps(findings,indent=2)+"\n"
    print(output)
    if args.report:
        path=root/args.report
        path.parent.mkdir(exist_ok=True,parents=True)
        path.write_text(output)
    failed=any(x["untracked_paths"] for x in violations)
    if args.strict:
        failed |= any(x["no_visible_numeric_evidence"] or x["no_visible_result_figures"] for x in violations)
    return 1 if failed else 0

if __name__=="__main__":
    sys.exit(main())
