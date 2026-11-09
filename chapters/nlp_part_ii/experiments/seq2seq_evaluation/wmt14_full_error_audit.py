#!/usr/bin/env python3
"""Post-hoc analysis of FULL WMT14 de-en MarianMT saved predictions.

Corpus SacreBLEU and chrF are taken from the upstream full evaluation;
sentence-level chrF recorded by that runner is analyzed separately.
"""
import argparse,json,hashlib,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--predictions",type=Path,required=True)
    p.add_argument("--original-metrics",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(x) for x in a.predictions.open() if x.strip()]
    original=json.loads(a.original_metrics.read_text())
    assert original["full_test"] and len(rows)==original["n_evaluated"]==3003
    for i,r in enumerate(rows):
        assert str(i)==str(r["id"]) and r["reference"].strip() and r["prediction"].strip()
    chrf=np.asarray([float(r["example_score"]) for r in rows])
    length=np.asarray([len(r["reference"].split()) for r in rows])
    assert np.all(np.isfinite(chrf)) and np.min(chrf)>=0 and np.max(chrf)<=100
    groups=[(1,10),(11,20),(21,30),(31,100000)]
    summary={"dataset":"WMT14 German–English official test","n":3003,
        "model":original["checkpoint"],"full_test":True,
        "reported_corpus_sacrebleu":original["metrics"]["sacrebleu_corpus"],
        "reported_corpus_chrf":original["metrics"]["chrf_corpus"],
        "mean_sentence_chrf":float(chrf.mean()),
        "prediction_sha256":hashlib.sha256(a.predictions.read_bytes()).hexdigest(),
        "by_reference_length":[],
        "limitations":["Post-hoc descriptive strata, not controlled ablation",
            "Mean sentence chrF and corpus chrF are distinct statistics"]}
    for lo,hi in groups:
        s=chrf[(length>=lo)&(length<=hi)]
        rng=np.random.default_rng(2026+lo)
        boot=rng.choice(s,(5000,len(s)),replace=True).mean(axis=1)
        summary["by_reference_length"].append({
            "stratum":f"{lo}-{hi if hi<100000 else '+'} words",
            "count":len(s),"mean_sentence_chrf":float(s.mean()),
            "ci95":np.quantile(boot,[.025,.975]).tolist()})
    (a.output/"full_wmt14_length_audit.json").write_text(json.dumps(summary,indent=2)+"\n")
    with (a.output/"full_wmt14_length_strata.csv").open("w",newline="") as stream:
        w=csv.writer(stream);w.writerow(["stratum","count","mean_sentence_chrf","lower95","upper95"])
        for r in summary["by_reference_length"]:
            w.writerow([r["stratum"],r["count"],r["mean_sentence_chrf"],*r["ci95"]])
    fig,ax=plt.subplots(figsize=(8,4.5))
    labels=[r["stratum"] for r in summary["by_reference_length"]]
    means=[r["mean_sentence_chrf"] for r in summary["by_reference_length"]]
    lows=[r["ci95"][0] for r in summary["by_reference_length"]]
    highs=[r["ci95"][1] for r in summary["by_reference_length"]]
    ax.bar(labels,means)
    ax.errorbar(range(4),means,yerr=[[m-l for m,l in zip(means,lows)],
                                    [h-m for m,h in zip(means,highs)]],
                fmt="none",capsize=4)
    ax.set(xlabel="Reference translation length",ylabel="Mean sentence chrF (0–100)",
           ylim=(0,100))
    fig.tight_layout();fig.savefig(a.output/"wmt14_length_dependence.pdf")
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
