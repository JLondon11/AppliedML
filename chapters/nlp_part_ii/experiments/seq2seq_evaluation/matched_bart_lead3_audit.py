#!/usr/bin/env python3
"""Match actual BART and Lead-3 predictions on the SAME 32 CNN/DailyMail examples.

The upstream BART runner stores sentence ROUGE-L F1 in `example_score`.
The upstream Lead-3 runner stores the identical rouge_score package's
rougeL F1 in `rougeL_f1`. Inputs MUST match exactly before any comparison.
This is exploratory first-32 performance, never a full test-set comparison.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read(path,limit=None):
    with open(path,encoding="utf-8") as f:
        rows=[json.loads(line) for line in f if line.strip()]
    return rows if limit is None else rows[:limit]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--bart",type=Path,required=True)
    p.add_argument("--lead3",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    bert=read(a.bart)
    lead=read(a.lead3,limit=len(bert))
    if len(bert)!=32 or len(lead)!=32:
        raise ValueError("This audit requires the frozen first 32 test articles")
    for i,(x,y) in enumerate(zip(bert,lead)):
        if x["source"]!=y["article"] or x["reference"]!=y["reference"]:
            raise ValueError(f"Mismatched source/reference at item {i}")
    bart=np.asarray([float(x["example_score"]) for x in bert])
    lead3=np.asarray([float(x["rougeL_f1"]) for x in lead])
    delta=bart-lead3
    rng=np.random.default_rng(421)
    boots=rng.choice(delta,(20000,len(delta)),replace=True).mean(axis=1)
    interval=np.quantile(boots,[.025,.975]).tolist()
    report={"dataset":"CNN/DailyMail 3.0.0",
        "subset":"FIRST 32 OFFICIAL TEST ITEMS (not random)",
        "paired_exact_match":True,"n":len(delta),
        "bart_mean_rougeL_f1":float(bart.mean()),
        "lead3_mean_rougeL_f1":float(lead3.mean()),
        "mean_paired_difference":float(delta.mean()),
        "percentile_bootstrap_ci95":interval,"seed":421,
        "n_bootstrap":20000,"bart_wins":int(sum(delta>0)),
        "lead3_wins":int(sum(delta<0)),"ties":int(sum(delta==0)),
        "scientific_limit":"Convenience subsample; NOT a full BART benchmark."}
    (a.output/"matched_comparison.json").write_text(json.dumps(report,indent=2)+"\n")
    with (a.output/"per_article_comparison.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["original_index","cnn_dailymail_id","bart_rougeL_f1",
                         "lead3_rougeL_f1","paired_delta"])
        for i,(x,b,l,d) in enumerate(zip(lead,bart,lead3,delta)):
            writer.writerow([i,x["id"],b,l,d])
    fig,ax=plt.subplots(figsize=(6.8,4.2))
    ax.bar(["BART-large-CNN","Lead-3"],[bart.mean(),lead3.mean()])
    ax.set_ylabel("Mean ROUGE-L F1");ax.set_ylim(0,.36)
    ax.set_title("Matched first 32 CNN/DailyMail test articles")
    fig.tight_layout();fig.savefig(a.output/"matched_rougeL_comparison.pdf")
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
