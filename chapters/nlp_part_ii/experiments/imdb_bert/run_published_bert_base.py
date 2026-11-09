#!/usr/bin/env python3
"""Execute the original BERT-base IMDb architecture on all official test reviews.

Uses published IMDb-fine-tuned checkpoint weights (NOT a new local fine-tune).
The checkpoint's model card provides external training provenance. All output
numbers are computed here from real held-out reviews and actual model logits.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from datasets import load_dataset
from torch.utils.data import DataLoader,Dataset
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,f1_score,
                             confusion_matrix,precision_recall_fscore_support)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--checkpoint",default="textattack/bert-base-uncased-imdb")
    p.add_argument("--revision",default="main")
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--max-length",type=int,default=128)
    a=p.parse_args()
    if a.batch_size<1 or a.max_length<1:raise ValueError("positive parameters")
    torch.manual_seed(42);torch.set_num_threads(2)
    ds=load_dataset("stanfordnlp/imdb")
    test=ds["test"]
    assert len(test)==25000 and len(ds["train"])==25000
    tokenizer=AutoTokenizer.from_pretrained(a.checkpoint,revision=a.revision,use_fast=True)
    model=AutoModelForSequenceClassification.from_pretrained(a.checkpoint,revision=a.revision)
    if model.config.num_labels!=2:raise ValueError("Binary IMDb checkpoint required")
    model.eval();device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    class Rows(Dataset):
        def __len__(self):return len(test)
        def __getitem__(self,i):return (i,test[i]["text"],test[i]["label"])
    def collate(rows):
        idx,texts,labels=zip(*rows)
        batch=tokenizer(list(texts),truncation=True,padding=True,
                        max_length=a.max_length,return_tensors="pt")
        return list(idx),batch,list(labels)
    loader=DataLoader(Rows(),batch_size=a.batch_size,shuffle=False,collate_fn=collate)
    out=[];gold=[];prediction=[];scores=[]
    with torch.no_grad():
        for ids,inputs,labels in loader:
            logits=model(**{k:v.to(device) for k,v in inputs.items()}).logits.cpu()
            proba=logits.softmax(-1)[:,1].numpy()
            labels_pred=logits.argmax(-1).numpy()
            for i,y,guess,prob in zip(ids,labels,labels_pred,proba):
                out.append({"id":str(i),"dataset":"stanfordnlp/imdb","split":"test",
                            "checkpoint":a.checkpoint,"revision":a.revision,
                            "y_true":int(y),"y_pred":int(guess),"prob_positive":float(prob)})
                gold.append(int(y));prediction.append(int(guess))
        # Full test must be covered once.
    if len(out)!=25000 or len(set(r["id"] for r in out))!=25000:
        raise ValueError("Incomplete/duplicate predictions")
    mat=confusion_matrix(gold,prediction,labels=[0,1])
    summary={"checkpoint":a.checkpoint,"revision":a.revision,
             "dataset":"stanfordnlp/imdb","checkpoint_training":"external published fine-tune, not retrained here",
             "sizes":{"official_train":25000,"official_test":len(test)},
             "max_length":a.max_length,"batch_size":a.batch_size,
             "test_metrics":{"accuracy":float(accuracy_score(gold,prediction)),
                 "balanced_accuracy":float(balanced_accuracy_score(gold,prediction)),
                 "macro_f1":float(f1_score(gold,prediction,average="macro")),
                 "confusion_matrix":mat.tolist()},
             "test_sha256":hashlib.sha256("\n".join(
                f'{r["label"]}\t{r["text"]}' for r in test).encode()).hexdigest(),
             "limitations":["Original checkpoint fine-tuned by outside authors",
                 "Not a new BERT-base training run or controlled architecture-only ablation",
                 "Maximum source length 128, matching public TextAttack model card",
                 "Model card does not prove complete training/test separation"]}
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/"results.json").write_text(json.dumps(summary,indent=2)+"\n")
    with (a.output/"predictions_test.jsonl").open("w") as f:
        for r in out:f.write(json.dumps(r)+"\n")
    fig,ax=plt.subplots(figsize=(5,4));ax.imshow(mat,cmap="Blues")
    for i in range(2):
        for j in range(2):ax.text(j,i,str(mat[i,j]),ha="center",va="center")
    ax.set(xticks=[0,1],yticks=[0,1],xlabel="Predicted",ylabel="True")
    fig.tight_layout();fig.savefig(a.output/"confusion_matrix.pdf");plt.close(fig)
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
