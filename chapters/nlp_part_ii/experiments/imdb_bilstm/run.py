#!/usr/bin/env python3
"""Train BiLSTM sentiment baseline on official IMDb with leak-free split.

22,500 official-train records, 2,500 stratified validation, 25,000 untouched
official test. Training and inference outputs are measured, never fabricated.
"""
import argparse,collections,hashlib,json,random,re
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from datasets import load_dataset
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"imdb_bert"))
from run import split_stratified

def tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[!?]",text.lower())

class BiLSTM(nn.Module):
    def __init__(self,vocab,dim=96,hidden=64):
        super().__init__()
        self.emb=nn.Embedding(vocab,dim,padding_idx=0)
        self.lstm=nn.LSTM(dim,hidden,batch_first=True,bidirectional=True)
        self.output=nn.Sequential(nn.Dropout(.2),nn.Linear(2*hidden,2))
    def forward(self,inputs,lengths):
        embedded=self.emb(inputs)
        packed=nn.utils.rnn.pack_padded_sequence(embedded,lengths.cpu(),
                    batch_first=True,enforce_sorted=False)
        _,(hidden,_)=self.lstm(packed)
        return self.output(torch.cat([hidden[-2],hidden[-1]],dim=1))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--max-length",type=int,default=128)
    p.add_argument("--vocab-size",type=int,default=20000)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    if min(a.epochs,a.batch_size,a.max_length,a.vocab_size)<2:
        raise ValueError("Positive training parameters required")
    torch.set_num_threads(2);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    ds=load_dataset("stanfordnlp/imdb")
    if len(ds["train"])!=25000 or len(ds["test"])!=25000:
        raise ValueError("Official IMDb split sizes changed")
    ids_train,ids_valid=split_stratified(ds["train"]["label"],seed=a.seed)
    splits={"train":ds["train"].select(ids_train),
            "validation":ds["train"].select(ids_valid),
            "test":ds["test"]}
    freq=collections.Counter(w for r in splits["train"] for w in tokenize(r["text"]))
    vocabulary={w:i+2 for i,(w,_) in enumerate(freq.most_common(a.vocab_size-2))}
    class Rows(Dataset):
        def __init__(self,records):self.rows=records
        def __len__(self):return len(self.rows)
        def __getitem__(self,i):
            row=self.rows[i]
            ids=[vocabulary.get(t,1) for t in tokenize(row["text"])[:a.max_length]]
            return (ids or [1],int(row["label"]),i)
    def batchify(examples):
        lengths=torch.tensor([len(e[0]) for e in examples],dtype=torch.long)
        ids=torch.zeros((len(examples),int(lengths.max())),dtype=torch.long)
        for row,(tokens,y,i) in enumerate(examples):
            ids[row,:len(tokens)]=torch.tensor(tokens)
        labels=torch.tensor([e[1] for e in examples],dtype=torch.long)
        indices=[e[2] for e in examples]
        return ids,lengths,labels,indices
    def make_loader(which,shuffle=False):
        return DataLoader(Rows(splits[which]),batch_size=a.batch_size,shuffle=shuffle,
            collate_fn=batchify)
    train=make_loader("train",True)
    valid=make_loader("validation")
    test=make_loader("test")
    model=BiLSTM(len(vocabulary)+2)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    criterion=nn.CrossEntropyLoss()
    def eval_loader(loader,train=False):
        model.train(train);loss_total=0;n=0;gold=[];pred=[];probs=[];indices=[]
        for x,lens,y,idx in loader:
            if train:optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train):
                logits=model(x,lens);loss=criterion(logits,y)
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                    optimizer.step()
            n+=len(y);loss_total+=float(loss.item())*len(y)
            gold+=y.tolist();pred+=logits.argmax(-1).detach().tolist()
            probs+=logits.softmax(-1)[:,1].detach().tolist();indices+=idx
        scores={"loss":loss_total/n,"accuracy":float(accuracy_score(gold,pred)),
                "macro_f1":float(f1_score(gold,pred,average="macro")),
                "confusion_matrix":confusion_matrix(gold,pred,labels=[0,1]).tolist()}
        return scores,gold,pred,probs,indices
    history=[];best=-1;best_state=None
    for epoch in range(1,a.epochs+1):
        tr,*_=eval_loader(train,True)
        va,*_=eval_loader(valid)
        history.append({"epoch":epoch,"train_loss":tr["loss"],
            "validation_loss":va["loss"],"validation_macro_f1":va["macro_f1"]})
        print(json.dumps(history[-1]),flush=True)
        if va["macro_f1"]>best:
            best=va["macro_f1"]
            best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    te,gold,pred,probs,indices=eval_loader(test)
    if len(indices)!=25000 or len(set(indices))!=25000:
        raise RuntimeError("Test predictions incomplete")
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/"predictions_test.jsonl").open("w") as f:
        for i,y,yhat,prob in zip(indices,gold,pred,probs):
            f.write(json.dumps({"id":str(i),"dataset":"stanfordnlp/imdb",
                "split":"test","model":"BiLSTM (96 embedding, 64 hidden each direction)",
                "y_true":y,"y_pred":yhat,"prob_positive":prob})+"\n")
    result={"model":"BiLSTM","dataset":"stanfordnlp/imdb","seed":a.seed,
        "architecture":{"embedding_dim":96,"hidden_per_direction":64,
           "vocab_size":len(vocabulary)+2,"max_length":a.max_length},
        "sizes":{k:len(v) for k,v in splits.items()},
        "split_sha256":{k:hashlib.sha256("\n".join(
            f'{r["label"]}\t{r["text"]}' for r in d).encode()).hexdigest()
                       for k,d in splits.items()},
        "test_metrics":te,"history":history}
    (a.output/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    fig,ax=plt.subplots(figsize=(6,4))
    for k in ["train_loss","validation_loss"]:
        ax.plot([t["epoch"] for t in history],[t[k] for t in history],label=k)
    ax.set(xlabel="Epoch",ylabel="Cross entropy");ax.legend()
    fig.tight_layout();fig.savefig(a.output/"training_curves.pdf");plt.close(fig)
    mat=np.array(te["confusion_matrix"]);fig,ax=plt.subplots(figsize=(5,4))
    ax.imshow(mat,cmap="Blues")
    for i in range(2):
        for j in range(2):ax.text(j,i,str(mat[i,j]),ha="center",va="center")
    ax.set(xticks=[0,1],yticks=[0,1],xlabel="Predicted",ylabel="True")
    fig.tight_layout();fig.savefig(a.output/"confusion_matrix.pdf");plt.close(fig)
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__":main()
