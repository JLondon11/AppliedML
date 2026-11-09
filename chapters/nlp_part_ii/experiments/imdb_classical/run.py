#!/usr/bin/env python3
"""IMDb official-data TF-IDF logistic regression benchmark with exact predictions."""
import hashlib,json
from pathlib import Path
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,
                             f1_score,precision_recall_fscore_support,confusion_matrix)
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"imdb_bert"))
from run import split_stratified

def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    ds=load_dataset("stanfordnlp/imdb")
    if len(ds["train"])!=25000 or len(ds["test"])!=25000:
        raise ValueError("Official split sizes changed")
    tr,va=split_stratified(ds["train"]["label"],seed=a.seed)
    train=ds["train"].select(tr)
    validation=ds["train"].select(va)
    test=ds["test"]
    pipe=Pipeline([("tfidf",TfidfVectorizer(ngram_range=(1,2),min_df=2,
                        max_features=100000,sublinear_tf=True,
                        strip_accents="unicode")),
                  ("clf",LogisticRegression(max_iter=500,C=2.,random_state=a.seed))])
    pipe.fit(train["text"],train["label"])
    scores={}
    for name,split in [("validation",validation),("test",test)]:
        pred=pipe.predict(split["text"]).astype(int)
        prob=pipe.predict_proba(split["text"])[:,1]
        gold=np.array(split["label"],dtype=int)
        scores[name]={
          "accuracy":float(accuracy_score(gold,pred)),
          "balanced_accuracy":float(balanced_accuracy_score(gold,pred)),
          "macro_f1":float(f1_score(gold,pred,average="macro")),
          "precision_recall_f1":{str(label):{"precision":float(x[0]),"recall":float(x[1]),
               "f1":float(x[2]),"support":int(x[3])} for label,x in zip([0,1],
               zip(*precision_recall_fscore_support(gold,pred,labels=[0,1])))} ,
          "confusion_matrix":confusion_matrix(gold,pred,labels=[0,1]).tolist()
        }
        if name=="test":
            a.output.mkdir(parents=True,exist_ok=True)
            with (a.output/"predictions_test.jsonl").open("w") as f:
                for i,(y,yhat,p1) in enumerate(zip(gold,pred,prob)):
                    f.write(json.dumps({"id":str(i),"dataset":"stanfordnlp/imdb",
                        "split":"test","model":"TF-IDF word (1,2)-gram + LogisticRegression",
                        "y_true":int(y),"y_pred":int(yhat),
                        "prob_positive":float(p1)})+"\n")
    result={"dataset":"stanfordnlp/imdb",
            "model":"TF-IDF word 1/2 gram + logistic regression",
            "seed":a.seed,"sizes":{"train":len(train),"validation":len(validation),
                                  "test":len(test)},
            "split_sha256":{k:hashlib.sha256("\n".join(
                 f'{r["label"]}\t{r["text"]}' for r in d).encode()).hexdigest()
                 for k,d in [("train",train),("validation",validation),("test",test)]},
            "metrics":scores,
            "hyperparameters":{"C":2,"max_features":100000,"min_df":2,
                               "max_iter":500,"ngram_range":[1,2]}}
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    mat=np.array(scores["test"]["confusion_matrix"])
    fig,ax=plt.subplots(figsize=(5,4))
    ax.imshow(mat,cmap="Blues")
    for i in range(2):
      for j in range(2):ax.text(j,i,str(mat[i,j]),ha="center",va="center")
    ax.set(xticks=[0,1],yticks=[0,1],xlabel="Predicted",ylabel="True")
    fig.tight_layout();fig.savefig(a.output/"confusion_matrix.pdf")
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
