#!/usr/bin/env python3
"""Real SciFact evidence-document retrieval; selects model on train only."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
def read(p):return [json.loads(z) for z in p.read_text().splitlines() if z.strip()]
def tops(X,Q):
    out=[]
    for i in range(0,Q.shape[0],32):
        scores=(Q[i:i+32]@X.T).toarray()
        out.extend([np.argsort(-r,kind="stable")[:10].tolist() for r in scores])
    return out
def bm25(x,q):
    n=x.shape[0];df=np.asarray((x>0).sum(axis=0)).ravel()
    idf=np.log(1+(n-df+.5)/(df+.5));length=np.asarray(x.sum(axis=1)).ravel()
    w=x.tocsr(copy=True).astype("float32");avg=length.mean()
    for i in range(n):
        a,b=w.indptr[i:i+2];v=w.data[a:b];normal=1.2*(.25+.75*length[i]/avg)
        w.data[a:b]=v*2.2/(v+normal)
    return tops(w.multiply(idf).tocsr(),(q>0).astype("float32"))
def metrics(items,pred,ids):
    out={}
    for k in [1,3,5,10]:
        out[f"recall@{k}"]=sum(bool(set(ids[t] for t in p[:k]) & set(c["evidence"]))
                                    for c,p in zip(items,pred))/len(items)
    out["mrr@10"]=sum(next((1/(j+1) for j,t in enumerate(p[:10])
                           if ids[t] in c["evidence"]),0.)
                          for c,p in zip(items,pred))/len(items)
    return out
def main():
    a=argparse.ArgumentParser();a.add_argument("--data",type=Path,required=True)
    a.add_argument("--output",type=Path,required=True);args=a.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    paths=[args.data/(s+".jsonl") for s in ["corpus","claims_train","claims_dev"]]
    docs=sorted(read(paths[0]),key=lambda x:int(x["doc_id"]))
    train=[x for x in read(paths[1]) if x.get("evidence")]
    dev=[x for x in read(paths[2]) if x.get("evidence")]
    assert (len(docs),len(train),len(dev))==(5183,505,188)
    ids=[str(d["doc_id"]) for d in docs]
    texts={"full":[d["title"]+" "+" ".join(d["abstract"]) for d in docs],
           "title":[d["title"] for d in docs]}
    query=[x["claim"] for x in train+dev]
    predictions={}
    for name,mode,analyzer,ngram in [
        ("tfidf_full","full","word",(1,2)),
        ("tfidf_title","title","word",(1,2)),
        ("tfidf_char","full","char_wb",(3,5))]:
        vec=TfidfVectorizer(analyzer=analyzer,ngram_range=ngram,
                            max_features=120000,sublinear_tf=True,dtype=np.float32)
        X=vec.fit_transform(texts[mode])
        predictions[name]=tops(X,vec.transform(query))
    v=CountVectorizer(max_features=90000,dtype=np.float32)
    X=v.fit_transform(texts["full"])
    predictions["bm25_full"]=bm25(X,v.transform(query))
    report={"dataset":"SciFact official latest release","n_corpus":len(docs),
     "n_train_annotated":len(train),"n_dev_annotated":len(dev),
     "split_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
     "metrics":{},"limits":["Retrieval only, no claim-verification inference",
      "No gold test labels are available in SciFact official test file",
      "Empty-gold claims excluded rather than counted as negatives"]}
    for name,pred in predictions.items():
        report["metrics"][name]={
          "train":metrics(train,pred[:len(train)],ids),
          "dev":metrics(dev,pred[len(train):],ids)}
    chosen=sorted(predictions,key=lambda k:(-report["metrics"][k]["train"]["recall@5"],k))[0]
    report["selected_on_train"]=chosen
    for split,items,offset in [("train",train,0),("dev",dev,len(train))]:
        with (args.output/f"predictions_{split}.jsonl").open("w") as f:
            for c,p in zip(items,predictions[chosen][offset:offset+len(items)]):
                ranking=[ids[z] for z in p];gold=list(c["evidence"])
                f.write(json.dumps({"id":c["id"],"claim":c["claim"],
                  "gold_evidence_doc_ids":gold,"predicted_doc_ids_top10":ranking,
                  "correct_top1":ranking[0] in gold,
                  "correct_top5":bool(set(ranking[:5])&set(gold))})+"\n")
    report["selection_rule"]="Highest train Recall@5, lexicographic tiebreak"
    (args.output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    names=list(predictions);loc=np.arange(len(names));fig,ax=plt.subplots(figsize=(8,4.4))
    for off,key in [(-.17,"recall@1"),(.17,"recall@5")]:
        ax.bar(loc+off,[report["metrics"][m]["dev"][key] for m in names],
               width=.34,label=key)
    ax.set_xticks(loc,names);ax.set_ylim(0,1);ax.set_ylabel("SciFact dev recall")
    ax.legend();fig.tight_layout();fig.savefig(args.output/"scifact_retrieval_comparison.pdf")
    print(json.dumps({"selected":chosen,"dev":report["metrics"][chosen]["dev"]}))
if __name__=="__main__":main()
