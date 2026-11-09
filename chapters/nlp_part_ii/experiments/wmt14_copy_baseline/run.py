#!/usr/bin/env python3
"""WMT14 German-to-English identity/copy baseline for full 3,003-pair test.

An intentionally weak sanity-check baseline: outputs original German unchanged.
Same official split and scoring library as the MarianMT full run. Does not
train a model or claim competitive translation performance.
"""
import argparse,hashlib,json
from pathlib import Path
from datasets import load_dataset
import sacrebleu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    ds=load_dataset("wmt/wmt14","de-en",split="test")
    if len(ds)!=3003:raise ValueError("Unexpected official WMT14 test size")
    german=[x["translation"]["de"] for x in ds]
    english=[x["translation"]["en"] for x in ds]
    predictions=list(german)
    scores={"sacrebleu_corpus":float(sacrebleu.corpus_bleu(predictions,[english]).score),
            "chrf_corpus":float(sacrebleu.corpus_chrf(predictions,[english]).score)}
    with (a.output/"predictions_test.jsonl").open("w",encoding="utf-8") as f:
        for i,(source,ref,pred) in enumerate(zip(german,english,predictions)):
            f.write(json.dumps({"id":str(i),"dataset":"wmt/wmt14 de-en",
              "split":"test","model":"copy-source sanity baseline","source":source,
              "reference":ref,"prediction":pred},ensure_ascii=False)+"\n")
    report={"dataset":"wmt/wmt14","config":"de-en","n_evaluated":len(ds),
      "full_test":True,"model":"copy-source sanity baseline (no training)",
      "metrics":scores,"sources_references_sha256":hashlib.sha256(
          "\n".join(x+"\t"+y for x,y in zip(german,english)).encode()).hexdigest(),
      "limitations":["Identity sanity baseline, not a competitive translation algorithm",
                     "Compared to MarianMT only on matched official test records"]}
    (a.output/"results.json").write_text(json.dumps(report,indent=2)+"\n")
    fig,ax=plt.subplots(figsize=(6,4))
    ax.bar(["SacreBLEU","chrF"],[scores["sacrebleu_corpus"],scores["chrf_corpus"]])
    ax.set(ylabel="Score (0–100)",ylim=(0,100))
    fig.tight_layout();fig.savefig(a.output/"copy_baseline_scores.pdf")
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
