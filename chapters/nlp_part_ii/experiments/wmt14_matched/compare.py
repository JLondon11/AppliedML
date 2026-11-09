#!/usr/bin/env python3
"""Verify and visualize full-test WMT14 MarianMT versus copy-source control."""
import argparse,csv,hashlib,json,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read_zip(path):
    with zipfile.ZipFile(path) as z:
        pred=next(n for n in z.namelist() if n.endswith("/predictions_test.jsonl"))
        meta=next(n for n in z.namelist() if n.endswith("/results.json"))
        return ([json.loads(t) for t in z.read(pred).decode().splitlines()],
                json.loads(z.read(meta)))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--marian",required=True,type=Path)
    p.add_argument("--control",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    m,mm=read_zip(a.marian);c,cm=read_zip(a.control)
    if len(m)!=3003 or len(c)!=3003:raise ValueError("Expected complete 3,003-pair WMT14 test")
    if any(x["source"]!=y["source"] or x["reference"]!=y["reference"] or str(x["id"])!=str(y["id"])
           for x,y in zip(m,c)):raise ValueError("Official test predictions not matched")
    if mm["sources_references_sha256"]!=cm["sources_references_sha256"]:
        raise ValueError("Test-set source/reference hash mismatch")
    report={"dataset":"WMT14 German-English official test","n_test":3003,
        "matched_rows":3003,"source_reference_sha256":mm["sources_references_sha256"],
        "marianmt":mm["metrics"],"copy_control":cm["metrics"],
        "difference":{k:mm["metrics"][k]-cm["metrics"][k] for k in mm["metrics"]},
        "input_archives_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [a.marian,a.control]},
        "limitations":["Copy-source is a sanity control, not competitive translation system",
                       "Model checkpoint was specified as revision 'main', not a pinned SHA",
                       "Corpus BLEU and chrF are corpus-level, not per-sentence significance tests"]}
    (a.output/"matched_translation_metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    with (a.output/"translation_examples.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["id","German","English reference","MarianMT","Copy-source"])
        for x,y in list(zip(m,c))[:24]:
            w.writerow([x["id"],x["source"],x["reference"],x["prediction"],y["prediction"]])
    fig,ax=plt.subplots(figsize=(7,4.4));xs=np.arange(2);width=.34
    keys=["sacrebleu_corpus","chrf_corpus"]
    ax.bar(xs-width/2,[mm["metrics"][k] for k in keys],width,label="MarianMT")
    ax.bar(xs+width/2,[cm["metrics"][k] for k in keys],width,label="Copy-source control")
    ax.set(xticks=xs,xticklabels=["Corpus SacreBLEU","Corpus chrF"],ylabel="Score (0–100)",
           ylim=(0,100),title="WMT14 de-en: 3,003 matched test pairs")
    ax.legend();fig.tight_layout()
    fig.savefig(a.output/"matched_full_wmt14_metrics.pdf")
    fig.savefig(a.output/"matched_full_wmt14_metrics.png",dpi=180)
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
