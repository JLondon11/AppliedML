#!/usr/bin/env python3
"""Small but real BERT/IMDb fine-tuning pilot, never a full benchmark.

Subsamples official IMDb train and test by deterministic stratified draws.
Full official 25k test remains reserved for the certification run.
"""
import argparse,json,hashlib,random
from pathlib import Path
import numpy as np
import torch
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from torch.utils.data import DataLoader,Dataset
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(path):
 t=pq.read_table(path,columns=["text","label"])
 return t.column("text").to_pylist(),np.asarray(t.column("label").to_pylist())
def subset(indices,labels,size,seed):
 if size>=len(indices):raise ValueError("Subsample must be strictly smaller")
 selection,_=train_test_split(np.asarray(indices),train_size=size,
         random_state=seed,stratify=np.asarray(labels)[np.asarray(indices)])
 return selection
class Rows(Dataset):
 def __init__(self,ids,text,labels):self.ids=ids;self.text=text;self.labels=labels
 def __len__(self):return len(self.ids)
 def __getitem__(self,i):idx=int(self.ids[i]);return idx,self.text[idx],int(self.labels[idx])
def main():
 parser=argparse.ArgumentParser()
 for name in ("train","test","output"):parser.add_argument("--"+name,type=Path,required=True)
 parser.add_argument("--model",default="bert-base-uncased")
 parser.add_argument("--seed",type=int,default=42)
 parser.add_argument("--train-count",type=int,default=600)
 parser.add_argument("--valid-count",type=int,default=200)
 parser.add_argument("--test-count",type=int,default=500)
 parser.add_argument("--epochs",type=int,default=1)
 parser.add_argument("--batch-size",type=int,default=8)
 args=parser.parse_args()
 if min(args.train_count,args.valid_count,args.test_count,args.epochs,args.batch_size)<1:
  raise ValueError("Counts and epochs must be positive")
 args.output.mkdir(parents=True,exist_ok=True)
 random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
 texts,y=load(args.train);texts_t,yt=load(args.test)
 assert (len(texts),len(texts_t))==(25000,25000)
 train_pool,valid= train_test_split(np.arange(len(y)),test_size=args.valid_count,
     random_state=args.seed,stratify=y)
 train=subset(train_pool,y,args.train_count,args.seed)
 test=subset(np.arange(len(yt)),yt,args.test_count,args.seed)
 assert not(set(train)&set(valid))
 tokenizer=AutoTokenizer.from_pretrained(args.model)
 model=AutoModelForSequenceClassification.from_pretrained(args.model,num_labels=2)
 model.to("cpu")
 def collate(batch):
  idx,x,labels=zip(*batch)
  payload=tokenizer(list(x),padding=True,truncation=True,max_length=128,
                    return_tensors="pt")
  payload["labels"]=torch.tensor(labels,dtype=torch.long)
  return idx,payload
 def loader(ids,txt,labels,shuffle=False):
  return DataLoader(Rows(ids,txt,labels),batch_size=args.batch_size,
                     shuffle=shuffle,collate_fn=collate,num_workers=0)
 opt=torch.optim.AdamW(model.parameters(),lr=2e-5)
 history=[]
 for epoch in range(args.epochs):
  model.train();n=0;total=0.
  for ids,batch in loader(train,texts,y,True):
   opt.zero_grad(set_to_none=True)
   loss=model(**batch).loss
   loss.backward()
   torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
   opt.step()
   n+=len(ids);total+=float(loss.item())*len(ids)
  model.eval();v=[];vy=[]
  with torch.no_grad():
   for ids,batch in loader(valid,texts,y):
    logits=model(**batch).logits
    v.extend(logits.argmax(-1).tolist());vy.extend(batch["labels"].tolist())
  h={"epoch":epoch+1,"train_cross_entropy":total/n,
     "validation_accuracy":float(accuracy_score(vy,v))}
  history.append(h);print("EPOCH",json.dumps(h),flush=True)
 model.eval();ids_out=[];actual=[];preds=[]
 with torch.no_grad():
  for ids,batch in loader(test,texts_t,yt):
   output=model(**batch).logits.argmax(-1).tolist()
   ids_out.extend([int(j) for j in ids]);actual.extend(batch["labels"].tolist());preds.extend(output)
 scores={"accuracy":float(accuracy_score(actual,preds)),
   "macro_f1":float(f1_score(actual,preds,average="macro")),
   "confusion_matrix":confusion_matrix(actual,preds,labels=[0,1]).tolist()}
 with (args.output/"predictions_test_subset.jsonl").open("w") as stream:
  for idx,real,guess in zip(ids_out,actual,preds):
   stream.write(json.dumps({"official_test_index":idx,"y_true":real,"y_pred":guess,
      "dataset":"stanfordnlp/imdb","checkpoint":args.model,"split":"official_test_stratified_subset"})+"\n")
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 report={"experiment":"BERT pilot, not certified full official IMDb experiment",
   "model":args.model,"train_count":len(train),"validation_count":len(valid),
   "official_test_subset_count":len(test),"official_full_test_count":len(yt),
   "seed":args.seed,"epochs":args.epochs,"history":history,"test_subset_metrics":scores,
   "sha256":{"train":sha(args.train),"test":sha(args.test)},
   "limitations":["Pilot uses a stratified subset of the official training split and test split",
   "Results must not be presented as full IMDb test benchmark performance"]}
 (args.output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
 fig,ax=plt.subplots();cm=np.asarray(scores["confusion_matrix"])
 ax.imshow(cm,cmap="Blues")
 for i in range(2):
  for j in range(2):ax.text(j,i,str(cm[i,j]),ha="center",va="center")
 ax.set(xticks=[0,1],yticks=[0,1],xlabel="Predicted label",ylabel="True label")
 fig.tight_layout();fig.savefig(args.output/"bert_pilot_confusion.pdf")
 print("TEST SUBSET",json.dumps(scores),flush=True)
if __name__=="__main__":main()
