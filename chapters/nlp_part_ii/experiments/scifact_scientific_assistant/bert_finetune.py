#!/usr/bin/env python3
"""Fine-tune a pretrained BERT-family classifier for SciFact evidence verification.

Real official SciFact train/dev; full-corpus unsupervised TF-IDF first-stage
retrieval. The fine-tuning uses only train labels and fixed epochs.
Output includes actual predictions, exact data hashes and scientific figures.
This is claim verification, not generative scientific question answering.
"""
import argparse,hashlib,json,random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score,confusion_matrix,accuracy_score
from sklearn.metrics.pairwise import linear_kernel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run import read_jsonl,gold_for_claim,download_data,build_dataset,data_text

LABELS=["NEI","SUPPORT","CONTRADICT"]
label_to_id={x:i for i,x in enumerate(LABELS)}
def choose_label(gold):
    values=set(gold.values())
    if not values:return "NEI"
    if len(values)>1:return None
    return next(iter(values))

def training_pairs(train,docs):
    examples=[];seen=set()
    for claim in train:
        gold=gold_for_claim(claim)
        for doc_id,label in gold.items():
            if doc_id not in docs:continue
            key=(claim["id"],doc_id)
            if key not in seen:
                examples.append((claim["claim"],data_text(docs[doc_id]),label))
                seen.add(key)
        for doc_id in claim.get("cited_doc_ids",[]):
            doc_id=str(doc_id)
            key=(claim["id"],doc_id)
            if doc_id in docs and key not in seen:
                examples.append((claim["claim"],data_text(docs[doc_id]),"NEI"))
                seen.add(key)
    if set(t[2] for t in examples)!=set(LABELS):
        raise ValueError("Training pairs do not contain all three classes")
    return examples

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",type=Path,default=Path("data"))
    p.add_argument("--output",type=Path,default=Path("results/scifact-bert"))
    p.add_argument("--download",action="store_true")
    p.add_argument("--checkpoint",default="prajjwal1/bert-tiny")
    p.add_argument("--revision",default="main")
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--max-length",type=int,default=256)
    p.add_argument("--lr",type=float,default=3e-5)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    if min(a.epochs,a.batch_size,a.max_length)<1:raise ValueError("Positive parameters required")
    if a.download:download_data(a.data_dir)
    corpus,train,dev,docs=build_dataset(a.data_dir)
    torch.set_num_threads(2)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    doc_ids=[str(d["doc_id"]) for d in corpus]
    vectorizer=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,
                               stop_words="english",dtype=np.float32,max_features=180000)
    doc_matrix=vectorizer.fit_transform([data_text(x) for x in corpus])
    top_ids=[]
    for c in dev:
        scores=linear_kernel(vectorizer.transform([c["claim"]]),doc_matrix).ravel()
        top_ids.append(doc_ids[int(scores.argmax())])
    pairs=training_pairs(train,docs)
    tokenizer=AutoTokenizer.from_pretrained(a.checkpoint,revision=a.revision,use_fast=True)
    model=AutoModelForSequenceClassification.from_pretrained(
        a.checkpoint,revision=a.revision,num_labels=3,
        id2label={i:v for i,v in enumerate(LABELS)},label2id=label_to_id).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr)
    class Rows(Dataset):
        def __len__(self):return len(pairs)
        def __getitem__(self,i):return pairs[i]
    def batchify(rows):
        x=tokenizer([t[0] for t in rows],[t[1] for t in rows],
                    truncation=True,padding=True,max_length=a.max_length,
                    return_tensors="pt")
        x["labels"]=torch.tensor([label_to_id[t[2]] for t in rows],dtype=torch.long)
        return x
    loader=DataLoader(Rows(),batch_size=a.batch_size,shuffle=True,collate_fn=batchify)
    history=[]
    for epoch in range(1,a.epochs+1):
        model.train();total=0;seen=0
        for item in loader:
            item={k:v.to(device) for k,v in item.items()}
            optimizer.zero_grad(set_to_none=True)
            loss=model(**item).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            n=len(item["labels"]);total+=float(loss.item())*n;seen+=n
        history.append({"epoch":epoch,"train_loss":total/seen})
        print(json.dumps(history[-1]),flush=True)
    model.eval()
    predictions=[]
    true_all=[];pred_all=[]
    mixed=0
    with torch.no_grad():
        for c,doc_id in zip(dev,top_ids):
            encoded=tokenizer(c["claim"],data_text(docs[doc_id]),
                              truncation=True,max_length=a.max_length,return_tensors="pt")
            encoded={k:v.to(device) for k,v in encoded.items()}
            probs=model(**encoded).logits.softmax(-1).squeeze(0).cpu().numpy()
            label=LABELS[int(probs.argmax())]
            gold=gold_for_claim(c)
            expected=choose_label(gold)
            if expected is None:mixed+=1
            else:
                true_all.append(expected);pred_all.append(label)
            predictions.append({"claim_id":c["id"],"claim":c["claim"],
                "retrieved_doc_id":doc_id,"retrieved_doc_title":docs[doc_id]["title"],
                "gold_evidence_doc_ids":sorted(gold),
                "reference_claim_label":expected,
                "model_label":label,
                "probabilities":{k:float(v) for k,v in zip(LABELS,probs)},
                "retrieved_doc_is_annotated_evidence":doc_id in gold})
    if len(true_all)<1:raise RuntimeError("No unambiguous held-out references")
    scores={"accuracy":float(accuracy_score(true_all,pred_all)),
            "macro_f1":float(f1_score(true_all,pred_all,labels=LABELS,average="macro",
                                     zero_division=0)),
            "top1_evidence_recall_on_positive_claims":float(
                np.mean([p["retrieved_doc_is_annotated_evidence"] for p in predictions
                        if p["gold_evidence_doc_ids"]])),
            "n_unambiguous_dev_claims":len(true_all),
            "n_ambiguous_dev_claims_excluded_from_label_metrics":mixed}
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/"predictions_dev.jsonl").open("w") as f:
        for rec in predictions:f.write(json.dumps(rec)+"\n")
    result={"dataset":"SciFact official train + held-out dev","corpus_n":len(corpus),
        "train_claims_n":len(train),"dev_claims_n":len(dev),
        "model_checkpoint":a.checkpoint,"revision":a.revision,"epochs":a.epochs,
        "batch_size":a.batch_size,"max_length":a.max_length,"seed":a.seed,
        "train_pairs_n":len(pairs),
        "train_nei_annotation":"unannotated cited documents are weak negatives",
        "retrieval":"TF-IDF word n-gram (1,2) on full SciFact abstract corpus",
        "scores":scores,"training":history,
        "input_sha256":{n:hashlib.sha256((a.data_dir/n).read_bytes()).hexdigest()
                        for n in ("corpus.jsonl","claims_train.jsonl","claims_dev.jsonl")},
        "limitations":["Not an official hidden-test benchmark result",
                       "Small pretrained BERT-family encoder, not a generative LLM",
                       "No rationale sentence selection or generative citation verification",
                       "Weak NEI labels from unannotated cited documents",
                       "Gold dev labels with contradictory source annotations excluded"]}
    (a.output/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    mat=confusion_matrix(true_all,pred_all,labels=LABELS)
    fig,ax=plt.subplots(figsize=(6,5));ax.imshow(mat,cmap="Blues")
    for i in range(3):
        for j in range(3):ax.text(j,i,str(mat[i,j]),ha="center",va="center")
    ax.set(xticks=range(3),yticks=range(3),xticklabels=LABELS,
           yticklabels=LABELS,xlabel="Predicted label",ylabel="Reference label")
    fig.tight_layout();fig.savefig(a.output/"scifact_bert_confusion.pdf");plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4))
    ax.plot([v["epoch"] for v in history],[v["train_loss"] for v in history],marker="o")
    ax.set(xlabel="Epoch",ylabel="Mean training cross-entropy")
    fig.tight_layout();fig.savefig(a.output/"scifact_bert_train_curve.pdf");plt.close(fig)
    print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__":main()
