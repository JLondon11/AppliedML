"""Figure 11 — Retrieval Quality as a Function of Top-k Depth.

Real-data experiment on BEIR SciFact test set (5,183 documents, 300 queries).
Lexical TF-IDF retrieval is evaluated against published qrels. All panels are
computed from the same ranked lists; no metric is hand-entered.
"""
from pathlib import Path
import json, time, zipfile, urllib.request
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

HERE=Path(__file__).resolve().parent
URL="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
DATA=HERE/"data"; DATA.mkdir(exist_ok=True)
ZIP=DATA/"scifact.zip"
if not ZIP.exists(): urllib.request.urlretrieve(URL,ZIP)
if not (DATA/"scifact").exists():
    with zipfile.ZipFile(ZIP) as z: z.extractall(DATA)
ROOT=DATA/"scifact"

import json as _json
corpus={}
with open(ROOT/"corpus.jsonl",encoding="utf8") as f:
    for line in f:
        x=_json.loads(line); corpus[x["_id"]]=(x.get("title","")+" "+x.get("text","")).strip()
queries={}
with open(ROOT/"queries.jsonl",encoding="utf8") as f:
    for line in f:
        x=_json.loads(line); queries[x["_id"]]=x["text"]
qrels={}
qdf=pd.read_csv(ROOT/"qrels"/"test.tsv",sep="\t")
for _,r in qdf.iterrows():
    if int(r["score"])>0: qrels.setdefault(str(r["query-id"]),set()).add(str(r["corpus-id"]))
qids=[q for q in queries if q in qrels]
docids=list(corpus); texts=[corpus[d] for d in docids]
vec=TfidfVectorizer(stop_words="english",ngram_range=(1,2),min_df=2,max_features=60000,dtype=np.float32)
X=normalize(vec.fit_transform(texts))
Q=normalize(vec.transform([queries[q] for q in qids]))
ks=np.array([1,2,3,5,10,20,50,100])
rec=np.zeros((len(qids),len(ks))); irrelevant=np.zeros_like(rec); toks=np.zeros_like(rec); latency=[]
rows=[]
for qi,qid in enumerate(qids):
    t0=time.perf_counter(); s=(Q[qi]@X.T).toarray().ravel(); order=np.argpartition(-s, min(100,len(s)-1))[:100]; order=order[np.argsort(-s[order])]; latency.append((time.perf_counter()-t0)*1000)
    rel=qrels[qid]
    for j,k in enumerate(ks):
        ids=[docids[i] for i in order[:k]]
        hits=sum(d in rel for d in ids); rec[qi,j]=hits/len(rel)
        irrelevant[qi,j]=1-hits/k
        toks[qi,j]=sum(len(corpus[d].split()) for d in ids)
    for rank,i in enumerate(order,1): rows.append((qid,rank,docids[i],float(s[i]),int(docids[i] in rel)))
mean_rec=rec.mean(0); sem_rec=rec.std(0,ddof=1)/np.sqrt(len(qids))
mean_noise=irrelevant.mean(0); mean_toks=toks.mean(0)
marginal=np.r_[mean_rec[0],np.diff(mean_rec)]
pd.DataFrame({"k":ks,"mean_recall":mean_rec,"sem_recall":sem_rec,"marginal_recall_gain":marginal,"mean_irrelevant_fraction":mean_noise,"mean_retrieved_whitespace_tokens":mean_toks}).to_csv(HERE/"figure_11_metrics.csv",index=False)
pd.DataFrame(rows,columns=["query_id","rank","doc_id","score","relevant"]).to_csv(HERE/"figure_11_ranked_results_top100.csv",index=False)
prov={"dataset":"BEIR SciFact","dataset_url":URL,"test_queries":len(qids),"corpus_documents":len(docids),"qrels_rows":int(len(qdf)),"retriever":"TF-IDF unigram+bigram cosine similarity","vocabulary_dimension":int(X.shape[1]),"k_values":ks.tolist(),"mean_query_retrieval_latency_ms":float(np.mean(latency)),"latency_note":"wall-clock sparse ranking time on GitHub Actions runner; environment-dependent","token_note":"whitespace-token count of retrieved document text; direct proxy for context/prompt burden, not tokenizer-specific billing","classification":"Case Study","section":"Case Study: Enterprise Retrieval-Augmented Generation","subsection":"Results and Error Analysis"}
(HERE/"figure_11_provenance.json").write_text(json.dumps(prov,indent=2)+"\n")

# Three scientific panels.
fig,axs=plt.subplots(1,3,figsize=(13.8,4.35))
ax=axs[0]; ax.plot(ks,mean_rec,marker="o",lw=1.5); ax.fill_between(ks,mean_rec-sem_rec,mean_rec+sem_rec,alpha=.16); ax.set_xscale("log"); ax.set_xlabel("Retrieved depth, k"); ax.set_ylabel("Mean evidence recall"); ax.set_ylim(0,1.02); ax.grid(False)
ax=axs[1]; ax.plot(ks,marginal,marker="o",lw=1.5); ax.axhline(0,lw=.7); ax.set_xscale("log"); ax.set_xlabel("Retrieved depth, k"); ax.set_ylabel("Marginal recall gain"); ax.grid(False)
ax=axs[2]; l1=ax.plot(ks,mean_toks,marker="o",lw=1.5,label="Context burden"); ax.set_xscale("log"); ax.set_xlabel("Retrieved depth, k"); ax.set_ylabel("Mean retrieved words"); ax2=ax.twinx(); l2=ax2.plot(ks,mean_noise,marker="s",lw=1.25,ls="--",label="Irrelevant fraction"); ax2.set_ylabel("Mean irrelevant fraction"); ax2.set_ylim(0,1.02); ax.grid(False)
for i,ax in enumerate(axs): ax.text(.5,-.23,f"({chr(97+i)})",transform=ax.transAxes,ha="center",va="top",fontsize=11)
fig.tight_layout(w_pad=2.2)
fig.savefig(HERE/"figure_11_retrieval_quality_topk.svg",bbox_inches="tight")
fig.savefig(HERE/"figure_11_retrieval_quality_topk.png",dpi=300,bbox_inches="tight")
plt.close(fig)
