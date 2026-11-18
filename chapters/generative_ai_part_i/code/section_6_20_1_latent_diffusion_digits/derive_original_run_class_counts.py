"""Derive actual 2,000-sample digit-generation count summary from one executed run.

Uses genuine run.py predictions.csv, not a fixed archived count table. The
archived Pass27 778/94 class counts remain a separate original experiment.
"""
import argparse,csv,json
from pathlib import Path
from collections import Counter
def derive(source,output,manifest=None):
    with Path(source).open(newline="",encoding="utf8") as stream:
        rows=list(csv.DictReader(stream))
    if len(rows)!=2000:raise ValueError("Expected 2,000 actual generated predictions")
    if manifest is not None:
        experiment=json.loads(Path(manifest).read_text())
        if experiment["n_generated_per_arm"]!=1000:
            raise ValueError("Actual source manifest prediction budget mismatch")
    counts=Counter()
    for row in rows:
        arm=row["arm"]
        if arm not in ("conditional","unconditional"):raise ValueError("Unknown model arm")
        digit=int(row["prompt_label"]);pred=int(row["classifier_prediction"])
        if digit not in range(10) or pred not in range(10):
            raise ValueError("Invalid requested or classifier label")
        if any(not 0<=float(row[f"p_{j}"])<=1 for j in range(10)):
            raise ValueError("Invalid original probability")
        if abs(sum(float(row[f"p_{j}"]) for j in range(10))-1)>1e-5:
            raise ValueError("Original output probabilities do not normalize")
        counts[(arm,digit,1)]+=int(digit==pred)
        counts[(arm,digit,0)]+=1
    summary=[]
    for arm in ("conditional","unconditional"):
        n=sum(counts[(arm,i,0)] for i in range(10))
        correct=sum(counts[(arm,i,1)] for i in range(10))
        if n!=1000:raise ValueError("Incompatible original run record count")
        if manifest is not None:
            key="class_conditional_consistency" if arm=="conditional" else "unconditional_consistency"
            if abs(correct/1000-experiment[key])>1e-12:
                raise ValueError("Manifest and original run classifier outcomes disagree")
        summary.append({"arm":arm,"label":"overall","correct":correct,"n":n})
        for digit in range(10):
            c=counts[(arm,digit,1)];N=counts[(arm,digit,0)]
            if N!=100:raise ValueError("Expected 100 actual generated samples per source class")
            summary.append({"arm":arm,"label":str(digit),"correct":c,"n":N})
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",newline="",encoding="utf8") as stream:
        w=csv.DictWriter(stream,fieldnames=["arm","label","correct","n"])
        w.writeheader();w.writerows(summary)
    return {x["arm"]:f"{x['correct']}/{x['n']}" for x in summary if x["label"]=="overall"}

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--predictions",required=True,type=Path)
    a.add_argument("--output",required=True,type=Path)
    a.add_argument("--manifest",type=Path)
    args=a.parse_args()
    print(json.dumps(derive(args.predictions,args.output,args.manifest),indent=2))
