#!/usr/bin/env python3
"""Train and test question-only evidence modality router on original TAT-QA."""
import argparse,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
LABELS=["table","table-text","text"]
def load(path):
 data=json.loads(path.read_text(encoding="utf-8"))
 return [{"id":q["uid"],"doc_id":str(d["table"]["uid"]),
          "question":q["question"],"source":q["answer_from"],
          "answer_type":q["answer_type"]}
         for d in data for q in d["questions"]]
def measures(y,p):
 return {"accuracy":float(accuracy_score(y,p)),
  "macro_f1":float(f1_score(y,p,labels=LABELS,average="macro")),
  "per_class_f1":{l:float(f1_score(y,p,labels=[l],average="macro")) for l in LABELS},
  "confusion_matrix":confusion_matrix(y,p,labels=LABELS).tolist()}
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--data",type=Path,required=True)
 p.add_argument("--output",type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 paths={s:a.data/("tatqa_"+s+".json") for s in ("train","dev","test")}
 data={s:load(path) for s,path in paths.items()}
 assert [len(data[s]) for s in ("train","dev","test")]==[13215,1668,1663]
 ids={s:{r["doc_id"] for r in data[s]} for s in data}
 for s in ids:
  for other in ids:
   if s<other and ids[s]&ids[other]:raise ValueError("Document leakage")
 v=TfidfVectorizer(ngram_range=(1,2),max_features=80000,min_df=2,sublinear_tf=True)
 X=v.fit_transform([z["question"] for z in data["train"]])
 V=v.transform([z["question"] for z in data["dev"]])
 T=v.transform([z["question"] for z in data["test"]])
 y={s:[z["source"] for z in data[s]] for s in data}
 models={"most_frequent":DummyClassifier(strategy="most_frequent"),
         "logreg_C0.5":LogisticRegression(C=.5,max_iter=1000),
         "logreg_C2":LogisticRegression(C=2,max_iter=1000)}
 validation={}
 for key,m in models.items():
  m.fit(X,y["train"]);validation[key]=measures(y["dev"],m.predict(V))
 chosen=sorted(models,key=lambda k:(-validation[k]["macro_f1"],k))[0]
 predictions=models[chosen].predict(T)
 result=measures(y["test"],predictions)
 with (a.output/"predictions_test.jsonl").open("w") as f:
  for x,pred in zip(data["test"],predictions):
   f.write(json.dumps({"id":x["id"],"document_id":x["doc_id"],
       "dataset":"TAT-QA official","split":"test","question":x["question"],
       "true_evidence_modality":x["source"],"predicted_evidence_modality":pred,
       "answer_type":x["answer_type"]})+"\n")
 report={"experiment":"tatqa_evidence_modality_router","dataset":"TAT-QA official",
  "model_choice_from_validation":chosen,"n_train":len(y["train"]),
  "n_validation":len(y["dev"]),"n_test":len(y["test"]),
  "label_distribution":{s:dict(Counter(y[s])) for s in y},
  "validation":validation,"test":result,
  "dataset_sha256":{s:hashlib.sha256(p.read_bytes()).hexdigest() for s,p in paths.items()},
  "limitations":["Question-only modality routing; not question answering",
                 "No user permissions or proprietary enterprise data tested"]}
 (a.output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
 fig,ax=plt.subplots(figsize=(7.2,5));cm=np.asarray(result["confusion_matrix"])
 ax.imshow(cm,cmap="Blues");ax.set_xticks(np.arange(3),LABELS,rotation=25)
 ax.set_yticks(np.arange(3),LABELS);ax.set(xlabel="Predicted modality",ylabel="True modality")
 for i in range(3):
  for j in range(3):ax.text(j,i,str(cm[i,j]),ha="center",va="center")
 fig.tight_layout();fig.savefig(a.output/"tatqa_modality_confusion.pdf")
 print(json.dumps({"selected":chosen,"test":result}))
if __name__=="__main__":main()
