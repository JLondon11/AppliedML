#!/usr/bin/env python3
"""Matched real BART versus Lead-3 evaluation; first 32 CNN/DailyMail test rows.

Requires two immutable published workflow artifact ZIPs. Rejects nonmatching
sources/references and does not claim full-test significance.
"""
import argparse,csv,hashlib,json,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest

def rows(zip_path,member):
    with zipfile.ZipFile(zip_path) as z:
        return [json.loads(x) for x in z.read(member).decode().splitlines() if x.strip()]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--bart",type=Path,required=True)
    p.add_argument("--lead3",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    b=rows(a.bart,"summarization/predictions_test.jsonl")
    l=rows(a.lead3,"lead3_full/predictions_test.jsonl")[:len(b)]
    if len(b)!=32 or len(l)!=32:raise ValueError("Expected first 32 aligned outputs")
    if not all(x["source"]==y["article"] and x["reference"]==y["reference"]
               for x,y in zip(b,l)):raise ValueError("Prediction source/reference mismatch")
    x=np.asarray([r["example_score"] for r in b])
    y=np.asarray([r["rougeL_f1"] for r in l])
    d=x-y
    rng=np.random.default_rng(20261109)
    boot=d[rng.integers(0,32,(50000,32))].mean(axis=1)
    wins=int((d>1e-12).sum());losses=int((d < -1e-12).sum())
    report={"dataset":"CNN/DailyMail 3.0.0","sample":"first 32 official test articles",
      "matched_sources":32,"matched_references":32,
      "mean_BART_ROUGE_L_F1":float(x.mean()),"mean_Lead3_ROUGE_L_F1":float(y.mean()),
      "paired_difference":float(d.mean()),
      "bootstrap95":np.quantile(boot,[.025,.975]).tolist(),
      "BART_wins":wins,"Lead3_wins":losses,"ties":32-wins-losses,
      "two_sided_sign_test_p":float(binomtest(wins,wins+losses,.5).pvalue),
      "sha256_bart_zip":hashlib.sha256(a.bart.read_bytes()).hexdigest(),
      "sha256_lead3_zip":hashlib.sha256(a.lead3.read_bytes()).hexdigest(),
      "limitations":["First 32 are not a random sample", "Not a complete BART test-set benchmark",
       "Original per-example F1 results computed with rouge_score (stemmer enabled)"]}
    (a.output/"matched_results.json").write_text(json.dumps(report,indent=2)+"\n")
    with (a.output/"paired_32_predictions.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["index","dataset_id","BART_ROUGEL","Lead3_ROUGEL","difference",
                                    "BART_prediction","Lead3_prediction"])
        for i,(r,s) in enumerate(zip(b,l)):
            w.writerow([i,s["id"],x[i],y[i],d[i],r["prediction"],s["prediction"]])
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.axhline(0,color="gray",linewidth=1);ax.bar(np.arange(32),d)
    ax.set(xlabel="Official CNN/DailyMail test index (0–31)",
           ylabel="BART minus Lead-3 ROUGE-L F1")
    fig.tight_layout()
    fig.savefig(a.output/"paired_rougeL_differences.pdf")
    fig.savefig(a.output/"paired_rougeL_differences.png",dpi=180)
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
