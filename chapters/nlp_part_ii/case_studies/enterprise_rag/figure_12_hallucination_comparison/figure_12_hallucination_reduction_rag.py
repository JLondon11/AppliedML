"""Figure 12 — Hallucination Reduction Through Retrieval Grounding.

Real controlled QA experiment on the public SciFact corpus/qrels. For each test
claim with a gold evidence document, compare:
  1. standalone generation: compact seq2seq model receives claim only;
  2. retrieval-grounded generation: same model receives claim + top retrieved text.
Unsupported claim rate is measured automatically by whether generated content
tokens are supported by the gold evidence text. Retrieval misses and generation
failures are retained as residual RAG failure categories.

This is a reproducible resource-conscious experiment, not a benchmark SOTA claim.
"""
from pathlib import Path
import json, zipfile, urllib.request, re
import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

HERE=Path(__file__).resolve().parent; DATA=HERE/"data"; DATA.mkdir(exist_ok=True)
URL="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"; ZIP=DATA/"scifact.zip"
if not ZIP.exists(): urllib.request.urlretrieve(URL,ZIP)
if not (DATA/"scifact").exists():
 with zipfile.ZipFile(ZIP) as z:z.extractall(DATA)
ROOT=DATA/"scifact"
corpus={}; queries={}
with open(ROOT/"corpus.jsonl",encoding="utf8") as f:
 for line in f:
  x=json.loads(line); corpus[x["_id"]]=(x.get("title","")+" "+x.get("text","")).strip()
with open(ROOT/"queries.jsonl",encoding="utf8") as f:
 for line in f:
  x=json.loads(line); queries[x["_id"]]=x["text"]
qdf=pd.read_csv(ROOT/"qrels"/"test.tsv",sep="\t")
qrels={}
for _,r in qdf.iterrows():
 if int(r["score"])>0:qrels.setdefault(str(r["query-id"]),set()).add(str(r["corpus-id"]))
qids=[q for q in queries if q in qrels][:120]  # deterministic first 120 judged claims
docids=list(corpus); texts=[corpus[d] for d in docids]
vec=TfidfVectorizer(stop_words="english",ngram_range=(1,2),min_df=2,max_features=50000,dtype=np.float32)
X=normalize(vec.fit_transform(texts)); Q=normalize(vec.transform([queries[q] for q in qids]))
MODEL="google/flan-t5-small"; tok=AutoTokenizer.from_pretrained(MODEL); model=AutoModelForSeq2SeqLM.from_pretrained(MODEL); model.eval()
stop=set(["the","a","an","and","or","of","to","in","is","are","was","were","for","on","with","that","this","as","by","from","it","be","has","have","had"])
def terms(s): return {w for w in re.findall(r"[a-z0-9]+",s.lower()) if len(w)>2 and w not in stop}
def gen(prompt):
 inp=tok(prompt,return_tensors="pt",truncation=True,max_length=512)
 with torch.no_grad(): out=model.generate(**inp,max_new_tokens=48,do_sample=False)
 return tok.decode(out[0],skip_special_tokens=True)
rows=[]
for qi,qid in enumerate(qids):
 s=(Q[qi]@X.T).toarray().ravel(); top=int(np.argmax(s)); retrieved=docids[top]
 gold_ids=qrels[qid]; gold_text=" ".join(corpus[d] for d in gold_ids if d in corpus)
 claim=queries[qid]
 standalone=gen("Answer the scientific claim concisely: "+claim)
 grounded=gen("Answer using only the evidence. Claim: "+claim+" Evidence: "+corpus[retrieved])
 gold_terms=terms(gold_text)
 def unsupported_rate(ans):
  at=terms(ans)
  return (sum(t not in gold_terms for t in at)/len(at)) if at else 1.0
 miss=int(retrieved not in gold_ids)
 u0=unsupported_rate(standalone); u1=unsupported_rate(grounded)
 generation_failure=int((not grounded.strip()) or u1>0.5)
 rows.append((qid,retrieved,miss,float(s[top]),standalone,grounded,u0,u1,generation_failure))
df=pd.DataFrame(rows,columns=["query_id","retrieved_doc","retrieval_miss","retrieval_score","standalone_answer","rag_answer","standalone_unsupported_rate","rag_unsupported_rate","generation_failure"])
df.to_csv(HERE/"figure_12_claim_level_results.csv",index=False)

rng=np.random.default_rng(1729)
def boot_mean(x,n=4000):
 x=np.asarray(x,float); ix=rng.integers(0,len(x),(n,len(x))); b=x[ix].mean(1); return float(x.mean()),np.quantile(b,[.025,.975]).tolist()
smean,sci=boot_mean(df.standalone_unsupported_rate); rmean,rci=boot_mean(df.rag_unsupported_rate)
miss_rate=float(df.retrieval_miss.mean())
genfail_rate=float(((df.generation_failure==1)&(df.retrieval_miss==0)).mean())
metrics={"standalone_unsupported_claim_rate":smean,"standalone_bootstrap_95ci":sci,"rag_unsupported_claim_rate":rmean,"rag_bootstrap_95ci":rci,"rag_retrieval_miss_rate":miss_rate,"rag_generation_failure_given_hit_population_rate":genfail_rate}
(HERE/"figure_12_metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
prov={"dataset":"BEIR SciFact","dataset_url":URL,"evaluated_claims":len(df),"generator":MODEL,"retriever":"TF-IDF unigram+bigram cosine top-1","support_metric":"fraction of non-stopword alphanumeric answer terms absent from concatenated gold evidence documents","bootstrap_seed":1729,"bootstrap_replicates":4000,"classification":"Case Study","section":"Case Study: Enterprise Retrieval-Augmented Generation","subsection":"Results and Error Analysis","caveat":"Lexical support is an automatic reproducible proxy for unsupported content, not human factuality adjudication. Results are protocol-specific and not a SOTA benchmark claim."}
(HERE/"figure_12_provenance.json").write_text(json.dumps(prov,indent=2)+"\n")

fig,axs=plt.subplots(1,2,figsize=(9.8,4.5))
ax=axs[0]; vals=[smean,rmean]; lo=[smean-sci[0],rmean-rci[0]]; hi=[sci[1]-smean,rci[1]-rmean]
ax.bar([0,1],vals,yerr=np.array([lo,hi]),capsize=4,width=.58); ax.set_xticks([0,1],["Standalone","Retrieval-grounded"]); ax.set_ylabel("Mean unsupported-term fraction"); ax.set_ylim(0,1)
ax=axs[1]; cats=["Retrieval miss","Generation failure\nafter retrieval hit"]; vv=[miss_rate,genfail_rate]; ax.bar([0,1],vv,width=.58); ax.set_xticks([0,1],cats); ax.set_ylabel("Fraction of evaluated claims"); ax.set_ylim(0,1)
for i,ax in enumerate(axs): ax.tick_params(direction="out"); ax.grid(False); ax.text(.5,-.23,f"({chr(97+i)})",transform=ax.transAxes,ha="center",va="top",fontsize=11)
fig.tight_layout(w_pad=2.6); fig.savefig(HERE/"figure_12_hallucination_reduction_rag.svg",bbox_inches="tight"); fig.savefig(HERE/"figure_12_hallucination_reduction_rag.png",dpi=300,bbox_inches="tight"); plt.close(fig)
