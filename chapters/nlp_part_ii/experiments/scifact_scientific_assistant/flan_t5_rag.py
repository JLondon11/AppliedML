#!/usr/bin/env python3
"""Real generative retrieval-augmented scientific claim verification on SciFact.

Actual FLAN-T5 predictions conditioned on first-stage retrieved SciFact abstracts.
No invented answers. The official development labels are held out.
"""
import argparse,hashlib,json,random,re
from pathlib import Path
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from sklearn.metrics.pairwise import linear_kernel
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run import download_data,build_dataset,gold_for_claim,data_text

VALID=["NEI","SUPPORT","CONTRADICT"]
ALL=VALID+["INVALID"]
def parse_label(raw):
    val=" ".join(raw.upper().split())
    if re.search(r"\b(CONTRADICT|REFUTE|REFUTED|FALSE)\b",val):return "CONTRADICT"
    if re.search(r"\b(SUPPORT|SUPPORTED|ENTAIL|ENTAILMENT|TRUE)\b",val):return "SUPPORT"
    if re.search(r"\b(NEI|NOT ENOUGH INFORMATION|INSUFFICIENT|UNKNOWN|UNCERTAIN)\b",val):
        return "NEI"
    return "INVALID"

def gold_label(claim):
    labels=set(gold_for_claim(claim).values())
    return "NEI" if not labels else next(iter(labels)) if len(labels)==1 else None

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",type=Path,default=Path("data"))
    p.add_argument("--output",type=Path,default=Path("results/scifact-generative"))
    p.add_argument("--download",action="store_true")
    p.add_argument("--checkpoint",default="google/flan-t5-small")
    p.add_argument("--revision",default="main")
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--max-source-length",type=int,default=384)
    p.add_argument("--max-new-tokens",type=int,default=12)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    if min(a.batch_size,a.max_source_length,a.max_new_tokens)<1:
        raise ValueError("Positive batch and token limits required")
    if a.download:download_data(a.data_dir)
    corpus,train,dev,docs=build_dataset(a.data_dir)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.set_num_threads(2)
    doc_ids=[str(d["doc_id"]) for d in corpus]
    vectorizer=TfidfVectorizer(ngram_range=(1,2),stop_words="english",
         sublinear_tf=True,max_features=180000,dtype=np.float32)
    matrix=vectorizer.fit_transform([data_text(d) for d in corpus])
    selected=[]
    for claim in dev:
        sims=linear_kernel(vectorizer.transform([claim["claim"]]),matrix).ravel()
        idx=int(sims.argmax())
        selected.append(doc_ids[idx])
    tokenizer=AutoTokenizer.from_pretrained(a.checkpoint,revision=a.revision)
    model=AutoModelForSeq2SeqLM.from_pretrained(a.checkpoint,revision=a.revision)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device);model.eval()
    prompts=[]
    for c,docid in zip(dev,selected):
        prompts.append(
            "Determine the relationship between a scientific claim and a retrieved "
            "abstract. Output exactly one label: SUPPORT, CONTRADICT, or NEI. "
            "Use NEI when the abstract does not establish the claim.\n"
            "Claim: "+c["claim"]+"\n"
            "Retrieved abstract: "+data_text(docs[docid])+"\nLabel:")
    raw_outputs=[]
    with torch.no_grad():
        for start in range(0,len(prompts),a.batch_size):
            encoded=tokenizer(prompts[start:start+a.batch_size],
                padding=True,truncation=True,max_length=a.max_source_length,
                return_tensors="pt")
            encoded={k:v.to(device) for k,v in encoded.items()}
            generated=model.generate(**encoded,do_sample=False,num_beams=1,
                                     max_new_tokens=a.max_new_tokens)
            raw_outputs.extend(tokenizer.batch_decode(generated,skip_special_tokens=True))
            print(f"predicted {len(raw_outputs)}/{len(prompts)}",flush=True)
    if len(raw_outputs)!=len(dev):
        raise RuntimeError("Incomplete generative inference")
    records=[];true=[];pred=[];ambiguous=0
    for c,docid,raw in zip(dev,selected,raw_outputs):
        gold=gold_for_claim(c);target=gold_label(c)
        status=parse_label(raw)
        if target is None:ambiguous+=1
        else:true.append(target);pred.append(status)
        evidence=docs[docid]
        records.append({"claim_id":c["id"],"claim":c["claim"],
            "reference_label":target,"raw_generated_verdict":raw,
            "parsed_verdict":status,
            "citation_doc_id":docid,
            "citation_title":evidence["title"],
            "citation_abstract_sentences":evidence["abstract"],
            "citation_text_verbatim_from_corpus":True,
            "citation_matches_annotated_evidence_doc":docid in gold,
            "annotated_evidence_document_ids":sorted(gold)})
    recognized=sum(x!="INVALID" for x in pred)
    metrics={"n_evaluated":len(true),"n_excluded_ambiguous":ambiguous,
        "recognized_label_rate":recognized/len(true),
        "accuracy":float(accuracy_score(true,pred)),
        "macro_f1":float(f1_score(true,pred,labels=VALID,average="macro",
                                 zero_division=0)),
        "citation_document_recall_at_1_for_annotated_claims":float(
            np.mean([r["citation_matches_annotated_evidence_doc"]
                     for r in records if r["annotated_evidence_document_ids"]]))}
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/"generative_predictions_dev.jsonl").open("w") as f:
        for r in records:f.write(json.dumps(r,ensure_ascii=False)+"\n")
    report={"model":a.checkpoint,"revision":a.revision,"dataset":"SciFact official",
        "corpus_n":len(corpus),"train_claims_n":len(train),"dev_claims_n":len(dev),
        "task":"zero-shot retrieval-grounded generated claim-verification verdict",
        "pipeline":"TF-IDF full corpus retrieval + FLAN-T5 conditioned generation",
        "seed":a.seed,"max_source_length":a.max_source_length,
        "max_new_tokens":a.max_new_tokens,"batch_size":a.batch_size,
        "metrics":metrics,"sha256":{f:hashlib.sha256((a.data_dir/f).read_bytes()).hexdigest()
            for f in ("corpus.jsonl","claims_train.jsonl","claims_dev.jsonl")},
        "limitations":[
            "No fine-tuning of FLAN-T5; zero-shot model inference only",
            "No independently generated explanatory rationale; model generates a verdict",
            "Cited documents are genuine, but citations may not contain gold evidence",
            "Only top-1 retrieved abstract supplied as model context",
            "No hidden SciFact test-set evaluation",
            "Parsing maps only recognized label variants, invalid outputs scored incorrect"]}
    (a.output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    cm=confusion_matrix(true,pred,labels=ALL)
    fig,ax=plt.subplots(figsize=(7,5))
    ax.imshow(cm,cmap="Blues")
    for i in range(len(ALL)):
        for j in range(len(ALL)):
            ax.text(j,i,str(cm[i,j]),ha="center",va="center")
    ax.set(xticks=range(len(ALL)),yticks=range(len(ALL)),
        xticklabels=ALL,yticklabels=ALL,xlabel="Generated label",
        ylabel="Reference label")
    fig.tight_layout();fig.savefig(a.output/"generated_label_confusion.pdf");plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)

if __name__=="__main__":main()
