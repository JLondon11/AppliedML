#!/usr/bin/env python3
"""Measured summarization or translation inference on complete labeled test splits.

Outputs are written only after executing an actual pretrained seq2seq model.
Full evaluations use original CNN/DailyMail or WMT14 test sets; sampled runs
are flagged as uncertifiable smoke tests.
"""
import argparse,hashlib,json,random
from pathlib import Path

def task_config(task):
    if task=="summarization":
        return {"dataset":"abisee/cnn_dailymail","config":"3.0.0",
                "model":"facebook/bart-large-cnn","source":"article","target":"highlights"}
    if task=="translation":
        return {"dataset":"wmt/wmt14","config":"de-en",
                "model":"Helsinki-NLP/opus-mt-de-en","source":"de","target":"en"}
    raise ValueError(task)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task",choices=["summarization","translation"],required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--checkpoint",default=None)
    p.add_argument("--revision",default="main")
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--source-max-length",type=int,default=512)
    p.add_argument("--max-new-tokens",type=int,default=128)
    p.add_argument("--num-beams",type=int,default=4)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--max-examples",type=int,default=None,
                   help="For smoke checks ONLY; results cannot be certified")
    a=p.parse_args()
    if min(a.batch_size,a.source_max_length,a.max_new_tokens,a.num_beams)<1:
        p.error("Positive generation parameters required")
    if a.max_examples is not None and a.max_examples<1:p.error("Positive max examples")
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sacrebleu
    from rouge_score import rouge_scorer
    cfg=task_config(a.task)
    checkpoint=a.checkpoint or cfg["model"]
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
    dataset=load_dataset(cfg["dataset"],cfg["config"],split="test")
    n_original=len(dataset)
    if a.max_examples is not None:dataset=dataset.select(range(min(a.max_examples,len(dataset))))
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,revision=a.revision)
    model=AutoModelForSeq2SeqLM.from_pretrained(checkpoint,revision=a.revision)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device);model.eval()
    references=[];predictions=[];sources=[]
    def data_fields(item):
        if a.task=="translation":
            return str(item["translation"]["de"]),str(item["translation"]["en"])
        return str(item[cfg["source"]]),str(item[cfg["target"]])
    with torch.no_grad():
        for offset in range(0,len(dataset),a.batch_size):
            batch=[data_fields(dataset[i]) for i in range(offset,min(offset+a.batch_size,len(dataset)))]
            x=[a[0] for a in batch];y=[a[1] for a in batch]
            tokenized=tokenizer(x,padding=True,truncation=True,
                                max_length=a.source_max_length,return_tensors="pt")
            tokenized={k:v.to(device) for k,v in tokenized.items()}
            tokens=model.generate(**tokenized,max_new_tokens=a.max_new_tokens,
                                  num_beams=a.num_beams,do_sample=False)
            output=tokenizer.batch_decode(tokens,skip_special_tokens=True)
            sources.extend(x);references.extend(y);predictions.extend(output)
            print(f"{offset+len(batch)}/{len(dataset)}",flush=True)
    if not predictions or len(predictions)!=len(references):
        raise RuntimeError("Incomplete model inference")
    if a.task=="summarization":
        scorer=rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"],use_stemmer=True)
        scores=[scorer.score(ref,pred) for ref,pred in zip(references,predictions)]
        metrics={k:float(np.mean([item[k].fmeasure for item in scores]))
                 for k in ("rouge1","rouge2","rougeL")}
        per_example=[float(row["rougeL"].fmeasure) for row in scores]
        metric_name="ROUGE-L F1"
    else:
        metrics={"sacrebleu_corpus":float(sacrebleu.corpus_bleu(predictions,[references]).score),
                 "chrf_corpus":float(sacrebleu.corpus_chrf(predictions,[references]).score)}
        per_example=[float(sacrebleu.sentence_chrf(pred,[ref]).score)
                     for pred,ref in zip(predictions,references)]
        metric_name="Sentence chrF (0–100)"
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/"predictions_test.jsonl").open("w",encoding="utf-8") as f:
        for i,(x,ref,pred,m) in enumerate(zip(sources,references,predictions,per_example)):
            f.write(json.dumps({"id":str(i),"dataset":cfg["dataset"],"split":"test",
                 "model":checkpoint,"version":a.revision,"task":a.task,
                 "source":x,"reference":ref,"prediction":pred,"example_score":m},
                 ensure_ascii=False)+"\n")
    result={"task":a.task,"dataset":cfg["dataset"],"config":cfg["config"],
            "checkpoint":checkpoint,"revision":a.revision,
            "source_max_length":a.source_max_length,"max_new_tokens":a.max_new_tokens,
            "num_beams":a.num_beams,"seed":a.seed,"n_original_test":n_original,
            "n_evaluated":len(predictions),"full_test":len(predictions)==n_original,
            "metrics":metrics,"sources_references_sha256":hashlib.sha256(
               "\n".join(x+"\t"+y for x,y in zip(sources,references)).encode()).hexdigest()}
    (a.output/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    fig,ax=plt.subplots(figsize=(7,4))
    ax.hist(per_example,bins=25)
    ax.set(xlabel=metric_name,ylabel="Number of held-out examples")
    fig.tight_layout();fig.savefig(a.output/"quality_distribution.pdf");plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    lengths=[(len(r.split()),len(q.split())) for r,q in zip(references,predictions)]
    ax.scatter([x for x,y in lengths],[y for x,y in lengths],alpha=.18,s=8)
    ax.set(xlabel="Reference word count",ylabel="Prediction word count")
    fig.tight_layout();fig.savefig(a.output/"output_length_scatter.pdf");plt.close(fig)
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
