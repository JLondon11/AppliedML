#!/usr/bin/env python3
"""Official SciFact scientific literature evidence retrieval and claim verification.

Uses official train/dev claims, full corpus, and genuine model predictions.
Dev is held out; official hidden test is NOT evaluated. No fabricated results.
"""
import argparse
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.metrics.pairwise import linear_kernel

URL="https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
LABELS=("NEI","SUPPORT","CONTRADICT")

def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def gold_for_claim(claim):
    labels={}
    for doc_id,ann in (claim.get("evidence") or {}).items():
        actual={x["label"] for x in ann}
        if len(actual)>1:
            raise ValueError("Conflicting labels for doc "+str(doc_id))
        labels[str(doc_id)]=next(iter(actual))
    return labels

def download_data(root):
    root.mkdir(parents=True,exist_ok=True)
    target=root/"data.tar.gz"
    if not target.exists():
        urllib.request.urlretrieve(URL,target)
    with tarfile.open(target,"r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():continue
            p=Path(member.name)
            if p.is_absolute() or ".." in p.parts:
                raise ValueError("Unsafe archive path")
            if p.name in ("corpus.jsonl","claims_train.jsonl","claims_dev.jsonl"):
                dest=root/p.name
                with archive.extractfile(member) as src,dest.open("wb") as dst:
                    while chunk:=src.read(1024*1024):dst.write(chunk)
    for name in ("corpus.jsonl","claims_train.jsonl","claims_dev.jsonl"):
        if not (root/name).is_file():raise RuntimeError("Missing "+name)

def build_dataset(root):
    corpus=read_jsonl(root/"corpus.jsonl")
    train=read_jsonl(root/"claims_train.jsonl")
    dev=read_jsonl(root/"claims_dev.jsonl")
    assert len(set(c["id"] for c in train)&set(c["id"] for c in dev))==0
    docs={str(d["doc_id"]):d for d in corpus}
    if len(docs)!=len(corpus):raise ValueError("Duplicate document ids")
    return corpus,train,dev,docs

def data_text(doc):
    return doc["title"]+" "+" ".join(doc["abstract"])

def train_classifier(train,docs,tfidf):
    # 3-way verifier: annotated evidence pairs + negative claim/doc pairs,
    # based solely on training claims. Negatives from train cited_doc_ids
    # are weak NEI labels and must be reported as such.
    x,y=[],[]
    for claim in train:
        gold=gold_for_claim(claim)
        positives=set(gold)
        for doc_id,label in gold.items():
            if doc_id in docs:
                x.append(claim["claim"]+" [SEP] "+data_text(docs[doc_id]))
                y.append(label)
        weak_nei=[str(i) for i in claim.get("cited_doc_ids",[])
                  if str(i) not in positives and str(i) in docs]
        for doc_id in weak_nei[:2]:
            x.append(claim["claim"]+" [SEP] "+data_text(docs[doc_id]))
            y.append("NEI")
    if set(y)!=set(LABELS):raise RuntimeError("Insufficient classes "+str(set(y)))
    xx=tfidf.transform(x)
    classifier=LogisticRegression(max_iter=1000,class_weight="balanced",random_state=42)
    classifier.fit(xx,y)
    return classifier,{"training_labeled_pairs":len(y),
                       "train_label_counts":{k:y.count(k) for k in LABELS},
                       "negative_label_source":"unannotated cited_doc_ids (weak NEI)"}

def run(root,out):
    corpus,train,dev,docs=build_dataset(root)
    doc_ids=[str(d["doc_id"]) for d in corpus]
    texts=[data_text(d) for d in corpus]
    retriever=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,
                              stop_words="english",dtype=np.float32,max_features=180000)
    matrix=retriever.fit_transform(texts)
    # Claim verification features trained on official TRAIN only.
    verifier_features=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,
                                     stop_words="english",min_df=2,
                                     max_features=140000,dtype=np.float32)
    training_pairs=[]
    for c in train:
        for id in gold_for_claim(c):
            if id in docs: training_pairs.append(c["claim"]+" [SEP] "+data_text(docs[id]))
        for id in c.get("cited_doc_ids",[]):
            if str(id) in docs:training_pairs.append(c["claim"]+" [SEP] "+data_text(docs[str(id)]))
    if not training_pairs:raise ValueError("Empty training pairs")
    verifier_features.fit(training_pairs)
    classifier,details=train_classifier(train,docs,verifier_features)
    topk=[1,3,5,10]
    relevant=[c for c in dev if gold_for_claim(c)]
    found={k:0 for k in topk}
    correct_label=[];pred_label=[];predictions=[]
    for c in dev:
        scores=linear_kernel(retriever.transform([c["claim"]]),matrix).reshape(-1)
        best=np.argpartition(scores,-max(topk))[-max(topk):]
        best=best[np.argsort(-scores[best])]
        ranked=[doc_ids[i] for i in best]
        gold=gold_for_claim(c)
        if gold:
            for k in topk:
                found[k]+=int(any(doc in gold for doc in ranked[:k]))
        top_id=ranked[0]
        top_doc=docs[top_id]
        # Prediction uses ONLY retrieved evidence; never gold during inference.
        pair=c["claim"]+" [SEP] "+data_text(top_doc)
        predicted=str(classifier.predict(verifier_features.transform([pair]))[0])
        # Single-doc claim label: claims with no supporting evidence are NEI.
        gold_status=(next(iter(set(gold.values()))) if gold else "NEI")
        if gold_status not in LABELS:
            raise ValueError("Unknown dev label")
        correct_label.append(gold_status)
        pred_label.append(predicted)
        predictions.append({
            "claim_id":c["id"],"claim":c["claim"],"gold_label":gold_status,
            "predicted_label":predicted,"retrieved_doc_id":top_id,
            "retrieved_doc_title":top_doc["title"],
            "retrieved_abstract":top_doc["abstract"],
            "top10_doc_ids":ranked,"gold_evidence_doc_ids":sorted(gold),
            "hit_at_1":bool(gold and top_id in gold),
            "evidence_source":"official SciFact corpus abstract"})
    scores={f"evidence_recall_at_{k}":found[k]/len(relevant) for k in topk}
    scores.update({
        "claim_label_macro_f1":float(f1_score(correct_label,pred_label,
                                            labels=list(LABELS),average="macro")),
        "claim_label_accuracy":float(np.mean(np.array(correct_label)==np.array(pred_label)))
    })
    metrics={
        "dataset":"SciFact official allenai/scifact release/latest",
        "data_source":URL,
        "corpus_n":len(corpus),"train_claims_n":len(train),
        "dev_claims_n":len(dev),"dev_evidence_positive_claims_n":len(relevant),
        "verification_model":"TF-IDF features + LogisticRegression trained on SciFact train",
        "retrieval_model":"word (1,2)-gram TF-IDF cosine, all corpus abstracts",
        "test_split":"official held-out dev (not hidden test)",
        "metrics":scores,"training":details,
        "limitations":[
            "Not official SciFact leaderboard evaluation; hidden test is not scored",
            "Weak NEI training pairs from unannotated citations may include false negatives",
            "One top-ranked abstract per claim; verifier lacks multi-document reasoning",
            "No pretrained transformer or generative large language model was executed",
            "Dev claim labels are not necessarily unique when documents disagree",
            "A document-level hit is not sentence-level rationale verification",
        ],
        "sha256":{file:hashlib.sha256((root/file).read_bytes()).hexdigest()
                 for file in ("corpus.jsonl","claims_train.jsonl","claims_dev.jsonl")}
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    with (out/"predictions_dev.jsonl").open("w") as f:
        for p in predictions:f.write(json.dumps(p)+"\n")
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot(topk,[scores[f"evidence_recall_at_{k}"] for k in topk],marker="o")
    ax.set(xlabel="Retrieved abstracts (k)",ylabel="Evidence document recall",
           ylim=(0,1),xticks=topk)
    fig.tight_layout();fig.savefig(out/"scifact_evidence_recall.pdf");plt.close(fig)
    cm=confusion_matrix(correct_label,pred_label,labels=list(LABELS))
    fig,ax=plt.subplots(figsize=(6,5))
    ax.imshow(cm,cmap="Blues")
    ax.set(xticks=range(3),yticks=range(3),xticklabels=LABELS,yticklabels=LABELS,
           xlabel="Predicted label",ylabel="Reference label")
    for i in range(3):
        for j in range(3):ax.text(j,i,str(cm[i,j]),ha="center",va="center")
    fig.tight_layout();fig.savefig(out/"scifact_verification_confusion.pdf");plt.close(fig)
    print(json.dumps(metrics,indent=2),flush=True)
    return metrics

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",type=Path,default=Path("data"))
    p.add_argument("--output",type=Path,default=Path("results/scifact"))
    p.add_argument("--download",action="store_true")
    a=p.parse_args()
    if a.download:download_data(a.data_dir)
    run(a.data_dir,a.output)

if __name__=="__main__":main()
