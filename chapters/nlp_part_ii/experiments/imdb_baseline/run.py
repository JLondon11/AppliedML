#!/usr/bin/env python3
"""Official IMDb: train TF-IDF logistic regression and MultinomialNB baselines.

Loads original 25k/25k official Stanford IMDb split. Selects configuration
on deterministic stratified 2.5k validation from train. Evaluates 25k official
test records ONCE after selection. Never represents this as BERT performance.
"""
import json,hashlib,argparse,csv
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,accuracy_score,balanced_accuracy_score,f1_score
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_parquet(p):
    import pyarrow.parquet as pq
    table=pq.read_table(p,columns=["text","label"])
    return table.column("text").to_pylist(),np.asarray(table.column("label").to_pylist())

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--test",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    train_text,y=load_parquet(args.train);test_text,yt=load_parquet(args.test)
    assert (len(train_text),len(test_text))==(25000,25000)
    assert (np.bincount(y).tolist(),np.bincount(yt).tolist())==([12500,12500],[12500,12500])
    trainids,validids=train_test_split(np.arange(25000),test_size=.1,random_state=42,
                                      shuffle=True,stratify=y)
    vec=TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=150000,
                        sublinear_tf=True,strip_accents="unicode")
    X=vec.fit_transform([train_text[i] for i in trainids])
    Xvalid=vec.transform([train_text[i] for i in validids])
    models={"LogReg_C1":LogisticRegression(C=1,max_iter=1000),
            "LogReg_C4":LogisticRegression(C=4,max_iter=1000),
            "MultinomialNB":MultinomialNB(alpha=1)}
    valstats={}
    for key,m in models.items():
        m.fit(X,y[trainids])
        pred=m.predict(Xvalid)
        valstats[key]={"accuracy":float(accuracy_score(y[validids],pred)),
                       "macro_f1":float(f1_score(y[validids],pred,average="macro"))}
        print("validation",key,valstats[key],flush=True)
    selected=sorted(models,key=lambda k:(-valstats[k]["macro_f1"],k))[0]
    # Final model uses entire official 25k train, selected solely by validation.
    vec=TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=150000,
                        sublinear_tf=True,strip_accents="unicode")
    Xall=vec.fit_transform(train_text)
    model=models[selected].__class__(**models[selected].get_params())
    model.fit(Xall,y)
    pred=model.predict(vec.transform(test_text))
    stats={"accuracy":float(accuracy_score(yt,pred)),
           "balanced_accuracy":float(balanced_accuracy_score(yt,pred)),
           "macro_f1":float(f1_score(yt,pred,average="macro")),
           "confusion_matrix":confusion_matrix(yt,pred,labels=[0,1]).tolist()}
    with (args.output/"predictions_test.jsonl").open("w") as f:
        for i,(real,guess) in enumerate(zip(yt,pred)):
            f.write(json.dumps({"id":i,"dataset":"stanfordnlp/imdb","split":"official_test",
                  "model":selected,"y_true":int(real),"y_pred":int(guess)})+"\n")
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    report={"dataset":"stanfordnlp/imdb","model_family":"TF-IDF linear/naive Bayes baseline, not BERT",
      "official_train_n":len(y),"official_test_n":len(yt),
      "validation_n":len(validids),"selection_only_on_validation":True,
      "training_refit_on_all_25000":True,"seed":42,"selected_model":selected,
      "validation_metrics":valstats,"test_metrics":stats,
      "dataset_sha256":{"train":sha(args.train),"test":sha(args.test)},
      "limitations":["Not a pretrained BERT experiment","Not a fine-tuned transformer",
        "Test evaluated once after validation model selection"]}
    (args.output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    fig,ax=plt.subplots(figsize=(6,5))
    cm=np.asarray(stats["confusion_matrix"])
    ax.imshow(cm,cmap="Blues")
    for i in range(2):
        for j in range(2):ax.text(j,i,str(cm[i,j]),ha="center",va="center")
    ax.set(xticks=[0,1],yticks=[0,1],xlabel="Predicted sentiment",ylabel="True sentiment")
    fig.tight_layout();fig.savefig(args.output/"imdb_confusion_matrix.pdf");plt.close(fig)
    print("FINAL",json.dumps(stats),flush=True)

if __name__=="__main__":main()
