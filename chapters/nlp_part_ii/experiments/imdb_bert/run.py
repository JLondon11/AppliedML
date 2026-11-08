#!/usr/bin/env python3
"""Train/evaluate BERT on official Stanford IMDb; never invent predictions."""
import argparse, hashlib, json, random
from pathlib import Path

def split_stratified(labels, fraction=.1, seed=42):
    import numpy as np
    rng=np.random.default_rng(seed); tr=[]; va=[]
    for cls in sorted(set(labels)):
        ids=rng.permutation([i for i,y in enumerate(labels) if y==cls]).tolist()
        n=round(len(ids)*fraction)
        if n<1 or n>=len(ids):raise ValueError("Insufficient samples for split")
        va.extend(ids[:n]);tr.extend(ids[n:])
    rng.shuffle(va);rng.shuffle(tr)
    assert not set(tr)&set(va)
    return tr,va

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--checkpoint",default="bert-base-uncased")
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--max-length",type=int,default=256)
    p.add_argument("--learning-rate",type=float,default=2e-5)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--local-only",action="store_true")
    a=p.parse_args()
    if min(a.epochs,a.batch_size,a.max_length)<1: p.error("Positive epochs/batch/length required")
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix
    from torch.utils.data import Dataset,DataLoader
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os
    if a.local_only:
        os.environ["HF_HUB_OFFLINE"]="1";os.environ["HF_DATASETS_OFFLINE"]="1"
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    data=load_dataset("stanfordnlp/imdb")
    if len(data["train"])!=25000 or len(data["test"])!=25000:
        raise ValueError("IMDb official split sizes changed; stop and investigate")
    tr,va=split_stratified(data["train"]["label"],seed=a.seed)
    splits={"train":data["train"].select(tr),"validation":data["train"].select(va),
            "test":data["test"]}
    tok=AutoTokenizer.from_pretrained(a.checkpoint,local_files_only=a.local_only,use_fast=True)
    model=AutoModelForSequenceClassification.from_pretrained(
        a.checkpoint,num_labels=2,local_files_only=a.local_only)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");model.to(device)
    class Rows(Dataset):
        def __init__(self,records):self.records=records
        def __len__(self):return len(self.records)
        def __getitem__(self,i):return self.records[i]
    def batchify(rows):
        e=tok([r["text"] for r in rows],truncation=True,padding=True,
              max_length=a.max_length,return_tensors="pt")
        e["labels"]=torch.tensor([r["label"] for r in rows],dtype=torch.long)
        return e
    def loader(name,shuffle=False):
        return DataLoader(Rows(splits[name]),batch_size=a.batch_size,
                          shuffle=shuffle,collate_fn=batchify)
    opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate)
    def evaluate(dl,train=False):
        model.train(train);total=0;n=0;gold=[];pred=[];probs=[]
        for row in dl:
            row={k:v.to(device) for k,v in row.items()}
            if train:opt.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train):
                out=model(**row)
                if train:
                    out.loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                    opt.step()
            count=len(row["labels"]);n+=count;total+=float(out.loss.item())*count
            gold.extend(row["labels"].cpu().tolist())
            pred.extend(out.logits.argmax(-1).cpu().tolist())
            probs.extend(out.logits.softmax(-1)[:,1].detach().cpu().tolist())
        scores={"loss":total/n,"accuracy":float(accuracy_score(gold,pred)),
                "balanced_accuracy":float(balanced_accuracy_score(gold,pred)),
                "macro_f1":float(f1_score(gold,pred,average="macro")),
                "confusion_matrix":confusion_matrix(gold,pred,labels=[0,1]).tolist()}
        return scores,gold,pred,probs
    best=-1;best_state=None;history=[]
    for epoch in range(1,a.epochs+1):
        train_scores,*_=evaluate(loader("train",True),True)
        valid_scores,*_=evaluate(loader("validation"))
        history.append({"epoch":epoch,"train_loss":train_scores["loss"],
                        "valid_loss":valid_scores["loss"],
                        "valid_macro_f1":valid_scores["macro_f1"]})
        print(json.dumps(history[-1]),flush=True)
        if valid_scores["macro_f1"]>best:
            best=valid_scores["macro_f1"]
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    scores,gold,pred,probs=evaluate(loader("test"))
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/"predictions_test.jsonl").open("w") as f:
        for i,(y,yhat,prob) in enumerate(zip(gold,pred,probs)):
            f.write(json.dumps({"id":str(i),"dataset":"stanfordnlp/imdb",
                     "split":"test","model":a.checkpoint,"version":"fine-tuned",
                     "y_true":y,"y_pred":yhat,"prob_positive":prob})+"\n")
    hashes={k:hashlib.sha256("\n".join(f'{r["label"]}\t{r["text"]}' for r in d)
               .encode()).hexdigest() for k,d in splits.items()}
    result={"model":a.checkpoint,"dataset":"stanfordnlp/imdb","seed":a.seed,
        "sizes":{k:len(v) for k,v in splits.items()},"split_sha256":hashes,
        "test_metrics":scores,"history":history,"device":str(device),
        "checkpoint_fine_tuned_on_train":True}
    (a.output/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    fig,ax=plt.subplots()
    for key in ["train_loss","valid_loss"]:
        ax.plot([r["epoch"] for r in history],[r[key] for r in history],label=key)
    ax.set(xlabel="Epoch",ylabel="Cross-entropy");ax.legend();fig.tight_layout()
    fig.savefig(a.output/"training_curves.pdf");plt.close(fig)
    fig,ax=plt.subplots();cm=np.asarray(scores["confusion_matrix"])
    ax.imshow(cm,cmap="Blues")
    for i in range(2):
        for j in range(2):ax.text(j,i,str(cm[i,j]),ha="center",va="center")
    ax.set(xticks=[0,1],yticks=[0,1],xlabel="Predicted",ylabel="True")
    fig.tight_layout();fig.savefig(a.output/"confusion_matrix.pdf");plt.close(fig)

if __name__=="__main__":main()
