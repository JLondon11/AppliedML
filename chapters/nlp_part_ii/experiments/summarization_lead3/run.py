#!/usr/bin/env python3
"""Lead-three-sentence baseline evaluated on all CNN/DailyMail 3.0.0 test articles."""
import argparse,json,hashlib,re
from pathlib import Path
import numpy as np
from datasets import load_dataset
from rouge_score import rouge_scorer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
def lead3(article):
    # Document the fixed rule: split on whitespace after terminal punctuation
    # when next token begins uppercase, and retain the first three segments.
    text=re.sub(r"\s+"," ",article.strip())
    parts=re.split(r"(?<=[.!?])\s+(?=[A-Z])",text)
    return " ".join(parts[:3])
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    data=load_dataset("abisee/cnn_dailymail","3.0.0",split="test")
    if len(data)!=11490:raise ValueError(f"Official split unexpected: {len(data)}")
    scorer=rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"],use_stemmer=True)
    scores=[];head=[]
    with (args.output/"predictions_test.jsonl").open("w") as f:
        for i,row in enumerate(data):
            pred=lead3(row["article"])
            values=scorer.score(row["highlights"],pred)
            rec={"id":row["id"],"dataset":"CNN/DailyMail 3.0.0",
                 "split":"official_test","model":"deterministic_lead3",
                 "article":row["article"],"reference":row["highlights"],
                 "prediction":pred,
                 "rouge1_f1":float(values["rouge1"].fmeasure),
                 "rouge2_f1":float(values["rouge2"].fmeasure),
                 "rougeL_f1":float(values["rougeL"].fmeasure)}
            f.write(json.dumps(rec,ensure_ascii=False)+"\n")
            scores.append([rec["rouge1_f1"],rec["rouge2_f1"],rec["rougeL_f1"]])
            if i<32:head.append(scores[-1])
    arr=np.asarray(scores);first=np.asarray(head)
    report={"dataset":"abisee/cnn_dailymail","version":"3.0.0",
        "model":"Lead3 deterministic extractive summarization baseline",
        "n_full_test":len(data),"n_first_32":len(first),
        "full_test_mean":{"rouge1":float(arr[:,0].mean()),
                          "rouge2":float(arr[:,1].mean()),
                          "rougeL":float(arr[:,2].mean())},
        "first_32_mean":{"rouge1":float(first[:,0].mean()),
                         "rouge2":float(first[:,1].mean()),
                         "rougeL":float(first[:,2].mean())},
        "rule":"First 3 regex-delimited sentence-like segments; punctuation boundary followed by uppercase initial",
        "limitations":["Not BART; not an abstractive language model",
                       "First-32 results are exploratory, not independent random sampling",
                       "BART sample comparison requires identical row IDs and ROUGE settings"]}
    (args.output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    fig,ax=plt.subplots(figsize=(7,4.2))
    keys=["ROUGE-1","ROUGE-2","ROUGE-L"]
    ax.bar(keys,[report["full_test_mean"][k] for k in ["rouge1","rouge2","rougeL"]])
    ax.set(ylabel="Mean ROUGE F1 (11,490 official test articles)",ylim=(0,1))
    fig.tight_layout();fig.savefig(args.output/"lead3_full_test_rouge.pdf")
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
