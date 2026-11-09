#!/usr/bin/env python3
"""Verify and compare predictions from two real IMDb experiments.
Usage:
  python paired_model_audit.py --a path/to/predictions_test.jsonl \
      --b path/to/predictions_test.jsonl --output measured
Works with ANY two named IMDb experiments using the same test IDs.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(path):
    result={}
    for line in path.read_text().splitlines():
        if not line.strip():continue
        x=json.loads(line)
        key=str(x["id"])
        if key in result:raise ValueError("Duplicate test id "+key)
        if x.get("split")!="test" or x.get("dataset")!="stanfordnlp/imdb":
            raise ValueError("Wrong dataset or split")
        result[key]=(int(x["y_true"]),int(x["y_pred"]))
    if len(result)!=25000:raise ValueError("Expected exactly 25,000 IMDb test predictions")
    return result

def run(a_path,b_path,output):
    a=load(a_path);b=load(b_path)
    if set(a)!=set(b):raise ValueError("Unmatched test identifiers")
    ids=sorted(a,key=int)
    if any(a[i][0]!=b[i][0] for i in ids):
        raise ValueError("Different reference labels; likely split mismatch")
    aok=np.array([a[i][0]==a[i][1] for i in ids])
    bok=np.array([b[i][0]==b[i][1] for i in ids])
    a_only=int(np.sum(aok &~bok));b_only=int(np.sum(~aok&bok))
    both=int(np.sum(aok&bok));neither=int(np.sum(~aok&~bok))
    p=float(binomtest(min(a_only,b_only),a_only+b_only,.5).pvalue)
    rng=np.random.default_rng(42)
    sampled=rng.multinomial(len(ids),[a_only/len(ids),b_only/len(ids),
                                       (both+neither)/len(ids)],size=30000)
    differences=(sampled[:,1]-sampled[:,0])/len(ids)
    confidence=np.quantile(differences,[.025,.975])
    report={"data":"stanfordnlp/imdb official test",
        "n_test":len(ids),"model_a_file":str(a_path),"model_b_file":str(b_path),
        "model_a_accuracy":float(aok.mean()),"model_b_accuracy":float(bok.mean()),
        "model_b_minus_a_accuracy":float(bok.mean()-aok.mean()),
        "paired_95_interval":list(map(float,confidence)),
        "discordance":{"a_only_correct":a_only,"b_only_correct":b_only,
                       "both_correct":both,"both_wrong":neither},
        "mcnemar_exact_two_sided_p":p}
    output.mkdir(parents=True,exist_ok=True)
    (output/"paired_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    fig,ax=plt.subplots(figsize=(7,4))
    vals=[float(aok.mean()),float(bok.mean())]
    ax.bar(["Model A","Model B"],np.array(vals)*100)
    ax.set(ylabel="Accuracy on official IMDb test (%)",ylim=(0,100))
    for i,v in enumerate(vals):ax.text(i,100*v+1,f"{v*100:.2f}%",ha="center")
    fig.tight_layout();fig.savefig(output/"paired_accuracy.pdf");plt.close(fig)
    print(json.dumps(report,indent=2))
if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a",type=Path,required=True)
    parser.add_argument("--b",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    run(args.a,args.b,args.output)
