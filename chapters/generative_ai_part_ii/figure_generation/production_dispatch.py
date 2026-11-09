"""Canonical Chapter 9 production renderer: copy only SHA-verified real-data figures.

No fallback is permitted. The frozen section-aware specs remain authoritative.
This validates experiment provenance; the full manuscript gate is separate.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve().parent
CHAPTER=HERE.parent
REPO=next(parent for parent in HERE.parents if (parent/"chapters").is_dir())
sys.path.insert(0,str(CHAPTER/"qa"))
from production_evidence import validate_production


def render(specs, figure=None, section=None, output_root=None):
    evidence=json.loads((CHAPTER/"qa/figure_evidence.json").read_text(encoding="utf-8"))
    by_number={int(record["figure_number"]):record for record in evidence["figures"]}
    if len(by_number)!=len(evidence["figures"]):
        raise RuntimeError("Duplicate figure evidence records")
    selected=[s for s in specs if (figure is None or int(s[0])==figure) and
              (section is None or s[1]==section)]
    if not selected:
        raise RuntimeError("No matching frozen Chapter 9 figure")
    # Validate all selected source files before writing any destination.
    resolved=[]
    for row in selected:
        number,sec,asset=row[:3]
        rec=by_number.get(int(number))
        if rec is None:
            raise RuntimeError(f"Chapter 9 Figure {number}: HOLD—no measured real-data evidence")
        errors=validate_production(rec,REPO,int(number))
        if errors:
            raise RuntimeError(f"Chapter 9 Figure {number}: HOLD—{' | '.join(errors)}")
        src=(REPO/rec["figure_path"]).resolve()
        dest_root=Path(output_root) if output_root else CHAPTER/"generated_real_data"
        dest=dest_root/Path(asset).name
        resolved.append((src,dest,int(number)))
    results=[]
    for src,dest,n in resolved:
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dest)
        results.append((n,dest))
        print(f"Chapter 9 Figure {n}: copied actual real-dataset experimental output to {dest}")
    return results


def main(specs):
    parser=argparse.ArgumentParser()
    parser.add_argument("--figure",type=int)
    parser.add_argument("--section")
    parser.add_argument("--output-root",type=Path)
    args=parser.parse_args()
    render(specs,args.figure,args.section,args.output_root)


if __name__=="__main__":
    raise SystemExit("Launch through figure_generation/generate_figures.py to load frozen inventory")
