"""Independent SciPy validation of 12 genuinely measured NEU classifier-arm/seed scores.

Cross-checks the standard-library source-based audit with scipy.stats and exact
binomial signs. The same 360 original held-out photographs are reused across
training seeds; no independent test-photo or factory generalization inference.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from scipy.stats import t, binomtest

SOURCE_SHA256="1f74ab0c8067612a072c2eff5554b971682ce851bd4c7ca5eb585fbb5f163d9e"
SEEDS=(20261106,20261107,20261108)
ARMS=("real_only","flip_augmentation","gan_augmented","real_repeat_augmented")

def verify(path: Path) -> dict:
    if hashlib.sha256(path.read_bytes()).hexdigest()!=SOURCE_SHA256:
        raise ValueError("The authentic 12-row source file SHA-256 has changed")
    with path.open(newline="",encoding="utf-8") as stream:
        rows=list(csv.DictReader(stream))
    if len(rows)!=12 or {(int(r["seed"]),r["arm"]) for r in rows}!={(s,a) for s in SEEDS for a in ARMS}:
        raise ValueError("Source seed/arm coverage mismatch")
    indexed={(int(r["seed"]),r["arm"]):r for r in rows}
    result={"original_dataset":"NEU-DET actual original steel-surface photographs",
            "source_sha256":SOURCE_SHA256,"source_records":12,
            "training_seeds":list(SEEDS),"same_original_test_photographs_per_seed":360,
            "paired_science":{}}
    for metric in ("test_accuracy","test_macro_f1"):
        result["paired_science"][metric]={}
        for comparator in ("real_only","real_repeat_augmented"):
            diffs=[float(indexed[s,"gan_augmented"][metric])-float(indexed[s,comparator][metric]) for s in SEEDS]
            if any(not math.isfinite(d) or not -1<=d<=1 for d in diffs):
                raise ValueError("Invalid observed effect")
            mean=statistics.mean(diffs);sd=statistics.stdev(diffs)
            critical=t.ppf(.975,2)
            interval=[float(mean-critical*sd/math.sqrt(3)),float(mean+critical*sd/math.sqrt(3))]
            neg=sum(d<0 for d in diffs)
            p=float(binomtest(neg,n=3,p=.5,alternative="greater").pvalue)
            if neg!=3 or p!=.125:raise ValueError("NEU actual negative seed result changed")
            result["paired_science"][metric][comparator]={
                "differences":diffs,"mean_difference":mean,"sd_between_seed_effects":sd,
                "descriptive_student_t_interval_95":interval,
                "exact_one_sided_sign_test_p":p}
    result["limits"]=[
        "Three training seeds, same 360 original real test photographs reused across all arms",
        "Descriptive Student t interval with only two degrees of freedom is assumption-sensitive",
        "Exact one-sided sign-test p 0.125 does not reject at 0.05",
        "Original per-photograph paired predictions unavailable in this CSV",
        "No new cross-factory, extra training-budget or deployment evidence"
    ]
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--csv",required=True,type=Path)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    data=json.dumps(verify(a.csv),indent=2)+"\n"
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(data,encoding="utf-8")
    print(data)
