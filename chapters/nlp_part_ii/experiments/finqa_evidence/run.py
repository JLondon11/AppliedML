#!/usr/bin/env python3
"""Official FinQA financial text/table evidence retrieval, train-selected baseline.

This is a supporting-evidence retrieval experiment, NOT mathematical question
answering and NOT enterprise authorization validation.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
def candidates(item):
    rows={f"text_{i}":str(x) for i,x in
          enumerate(item.get("pre_text",[])+item.get("post_text",[]))}
    table=item.get("table",[])
    if table:
        for i,row in enumerate(table):
            rows[f"table_{i}"]=" ; ".join(f"{h} : {v}" for h,v in zip(table[0],row))
    return rows
def rank(query,data,mode):
    keys=list(data)
    if not keys:return []
    vectorizer=(TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,
                              token_pattern=r"(?u)\b\w+\b")
                if mode=="word" else
                TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),max_features=12000))
    try:
        m=vectorizer.fit_transform(list(data.values())+[query])
        score=(m[:-1]@m[-1].T).toarray().ravel()
        return [keys[i] for i in np.argsort(-score,kind="stable")[:10]]
    except ValueError:return keys[:10]
def evaluate(items,mode,save=False):
    stats={k:[] for k in ["hit@1","hit@3","hit@5","all_gold@5","mrr@10"]}
    results=[];absent=0
    for item in items:
        gold=set(item["qa"]["gold_inds"])
        source=candidates(item);absent+=not gold.issubset(source)
        pred=rank(item["qa"]["question"],source,mode)
        found=[j+1 for j,key in enumerate(pred) if key in gold]
        for k in (1,3,5):stats[f"hit@{k}"].append(int(any(v<=k for v in found)))
        stats["all_gold@5"].append(int(bool(gold) and gold.issubset(pred[:5])))
        stats["mrr@10"].append(1/min(found) if found else 0.)
        if save:
            results.append({"id":item["id"],"dataset":"FinQA official",
              "question":item["qa"]["question"],"answer_reference":item["qa"]["answer"],
              "gold_evidence_ids":sorted(gold),"predicted_evidence_ids_top10":pred,
              "gold_ids_present_in_candidates":gold.issubset(source),
              "gold_top1":int(bool(found) and min(found)==1),
              "gold_top5":int(bool(found) and min(found)<=5)})
    return {k:float(np.mean(v)) for k,v in stats.items()},results,int(absent)
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    splits={s:json.loads((args.data/f"finqa_{s}.json").read_text())
            for s in ["train","dev","test"]}
    assert [len(splits[s]) for s in ["train","dev","test"]]==[6251,883,1147]
    scores={method:evaluate(splits["train"][:800],method)[0] for method in ["word","char"]}
    method=sorted(scores,key=lambda k:(-scores[k]["hit@5"],k))[0]
    report={"dataset":"FinQA official","selected_on_train":method,
      "selection_n":800,"selection_scores":scores,
      "counts":{k:len(v) for k,v in splits.items()},
      "dataset_sha256":{s:hashlib.sha256((args.data/f"finqa_{s}.json").read_bytes()).hexdigest()
                        for s in splits},"evaluation":{},"limitations":[
       "Evidence retrieval only, no arithmetic-answer correctness",
       "No validation of user roles, document permissions or proprietary enterprise data",
       "Method selected exclusively on first 800 training examples"]}
    for split in ["dev","test"]:
        measures,pred,missing=evaluate(splits[split],method,True)
        report["evaluation"][split]={**measures,"gold_reference_missing_candidate_n":missing}
        with (args.output/f"predictions_{split}.jsonl").open("w") as f:
            for row in pred:f.write(json.dumps(row,ensure_ascii=False)+"\n")
    (args.output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    fig,ax=plt.subplots(figsize=(7.5,4.3));labels=["hit@1","hit@3","hit@5","all_gold@5"]
    loc=np.arange(len(labels))
    for offset,split in [(-.175,"dev"),(.175,"test")]:
        ax.bar(loc+offset,[report["evaluation"][split][k] for k in labels],
               width=.35,label=split)
    ax.set_xticks(loc,labels);ax.set_ylim(0,1);ax.set_ylabel("Fraction of examples")
    ax.legend();fig.tight_layout();fig.savefig(args.output/"finqa_evidence_retrieval.pdf")
    print(json.dumps({"method":method,"test":report["evaluation"]["test"]}))
if __name__=="__main__":main()
