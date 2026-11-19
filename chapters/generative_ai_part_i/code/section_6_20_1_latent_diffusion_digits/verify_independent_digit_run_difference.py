"""Audit two genuinely executed digit-diffusion class-level results without conflating runs.

Archived Pass27 and successful separate CI run 44209629586 used the same real
sklearn handwritten-digit source and method, but fitted separate models.
Their original generated output counts should not be forced to bytewise match.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def load_counts(path: Path) -> dict:
    with path.open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    keys={(arm,str(i)) for arm in ("conditional","unconditional") for i in range(10)}
    keys|={(arm,"overall") for arm in ("conditional","unconditional")}
    if len(rows)!=22 or len({(r.get("arm"),r.get("label")) for r in rows})!=22:
        raise ValueError("Duplicated or missing original executed-run records")
    values={}
    for r in rows:
        key=(r.get("arm"),r.get("label"))
        if key not in keys:
            raise ValueError("Unknown original class or model arm")
        correct=int(r["correct"]);n=int(r["n"])
        if n!=(1000 if key[1]=="overall" else 100) or not 0<=correct<=n:
            raise ValueError("Invalid observed-run class denominator or count")
        values[key]=correct
    if set(values)!=keys:
        raise ValueError("Missing original class")
    for arm in ("conditional","unconditional"):
        if sum(values[arm,str(i)] for i in range(10))!=values[arm,"overall"]:
            raise ValueError("Source overall count does not reconcile")
    return values


def reconcile(archived: Path, current: Path) -> dict:
    a=load_counts(archived);b=load_counts(current)
    changed=[{"arm":arm,"digit":i,"archive_correct":a[arm,str(i)],
              "current_correct":b[arm,str(i)],
              "current_minus_archive":b[arm,str(i)]-a[arm,str(i)]}
             for arm in ("conditional","unconditional") for i in range(10)
             if a[arm,str(i)]!=b[arm,str(i)]]
    result={
      "archive":"Pass27 saved generated model classifier outputs",
      "current":"Separately executed original-data CI run 44209629586",
      "archive_conditional_correct":a["conditional","overall"],
      "current_conditional_correct":b["conditional","overall"],
      "archive_unconditional_correct":a["unconditional","overall"],
      "current_unconditional_correct":b["unconditional","overall"],
      "class_differences":changed,
      "scientific_interpretation":"Different genuine model executions; source result changes must be reported, never forced to the prior seed output",
      "limits":["Source-level aggregate comparison only","Does not independently reload the full 2000 raw probability records",
                "No cross-generator-seed uncertainty justified by two nonmatched source environments",
                "Does not establish Stable Diffusion or natural-image generator performance"]
    }
    if not (result["archive_conditional_correct"]==778 and result["current_conditional_correct"]==779
            and result["archive_unconditional_correct"]==result["current_unconditional_correct"]==94
            and changed==[{"arm":"conditional","digit":8,"archive_correct":66,
                          "current_correct":67,"current_minus_archive":1}]):
        raise ValueError("Unexpected new run drift; requires new independent source audit")
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--archived",type=Path,required=True)
    p.add_argument("--current",type=Path,required=True)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    result=json.dumps(reconcile(a.archived,a.current),indent=2)+"\n"
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(result,encoding="utf-8")
    print(result)
if __name__=="__main__":
    main()
