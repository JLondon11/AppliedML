"""Figure 10 — Enterprise Retrieval-Augmented Generation Architecture.

Scientific computational/data-flow rendering instantiated from executable
retrieval + reranking + grounded prompt assembly. No performance values are
invented. The diagram records tensor/data dimensions from the actual run.
"""
from pathlib import Path
import json, numpy as np, matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
HERE=Path(__file__).resolve().parent

docs=[
("D1","Security policy","Production access requires phishing-resistant MFA and quarterly access review."),
("D2","Retention policy","Customer support transcripts are retained for 30 days unless a legal hold applies."),
("D3","Incident response","Critical incidents require paging the on-call security lead and opening an incident record."),
("D4","Travel policy","International travel requires manager approval before booking."),
("D5","Data handling","Restricted customer data must not be copied into unapproved external systems."),
("D6","Access handbook","Privileged production roles are granted through the access-management workflow.")
]
query="What authentication is required for production access?"
texts=[d[2] for d in docs]
vec=TfidfVectorizer(ngram_range=(1,2),stop_words="english")
X=vec.fit_transform(texts); q=vec.transform([query])
scores=cosine_similarity(q,X)[0]
rank=np.argsort(-scores); topk=rank[:3]
# deterministic lexical reranking adds exact query-token coverage to semantic score
qt=set(vec.build_analyzer()(query))
rerank=[]
for i in topk:
 dt=set(vec.build_analyzer()(texts[i]))
 coverage=len(qt & dt)/max(1,len(qt))
 rerank.append((0.75*float(scores[i])+0.25*coverage,int(i)))
rerank=sorted(rerank,reverse=True)
selected=[i for _,i in rerank[:2]]
evidence=[docs[i] for i in selected]
prompt="QUESTION: "+query+"\nEVIDENCE:\n"+"\n".join(f"[{d[0]}] {d[2]}" for d in evidence)
answer="Production access requires phishing-resistant MFA [D1]."
prov={"query":query,"documents":[{"id":d[0],"title":d[1],"text":d[2]} for d in docs],
      "retrieval":"TF-IDF unigram+bigram cosine similarity","vocabulary_dimension":int(X.shape[1]),
      "top_k":3,"retrieval_scores":{docs[i][0]:float(scores[i]) for i in rank},
      "reranked_ids":[docs[i][0] for _,i in rerank],"prompt":prompt,
      "grounded_response":answer,"evidence_links":[docs[i][0] for i in selected],
      "note":"Executable miniature enterprise corpus used to instantiate the architecture; figure makes no benchmark-performance claim."}
(HERE/"figure_10_numerical_provenance.json").write_text(json.dumps(prov,indent=2)+"\n")

# Scientific data-flow: typed states and measured dimensions, not decorative infographic.
fig,ax=plt.subplots(figsize=(13.8,5.2)); ax.set_xlim(-.5,7.6); ax.set_ylim(-1.55,2.0); ax.axis("off")
x=[0,1.15,2.35,3.55,4.75,5.95,7.1]
names=["Query","Query vector","Hybrid/search\nindex","Top-k evidence","Reranked\nevidence","Grounded prompt","Generated\nresponse"]
dims=["text",f"1 × {X.shape[1]}",f"{len(docs)} × {X.shape[1]}","k = 3","k = 2",f"{len(prompt)} chars","text + citations"]
for i,(xx,n,d) in enumerate(zip(x,names,dims)):
 ax.plot([xx,xx],[.15,.78],lw=8,alpha=.15,solid_capstyle="butt")
 ax.text(xx,.91,n,ha="center",va="bottom",fontsize=9)
 ax.text(xx,.02,d,ha="center",va="top",fontsize=8)
 if i<len(x)-1: ax.annotate("",xy=(x[i+1]-.14,.46),xytext=(xx+.14,.46),arrowprops=dict(arrowstyle="->",lw=1.05))
# non-parametric enterprise knowledge enters retrieval
ax.text(2.35,1.72,"Enterprise document corpus",ha="center",fontsize=9)
ax.text(2.35,1.48,f"N = {len(docs)} documents",ha="center",fontsize=8)
ax.annotate("",xy=(2.35,.83),xytext=(2.35,1.38),arrowprops=dict(arrowstyle="->",lw=1.0))
# parametric knowledge enters generation separately
ax.text(7.1,1.72,"Parametric model state",ha="center",fontsize=9)
ax.text(7.1,1.48,"fixed pretrained parameters",ha="center",fontsize=8)
ax.annotate("",xy=(7.1,.83),xytext=(7.1,1.38),arrowprops=dict(arrowstyle="->",lw=1.0))
# evidence-link retention
ax.plot([3.55,7.1],[-.72,-.72],lw=.9,ls="--")
ax.annotate("",xy=(7.1,-.72),xytext=(3.55,-.72),arrowprops=dict(arrowstyle="->",lw=.9))
ax.text(5.3,-.92,"evidence identifiers retained through prompt → response",ha="center",fontsize=8)
# scientifically relevant failure sensitivities, compact axis-like annotations
ax.text(2.95,-1.34,"freshness / chunking / retrieval",ha="center",fontsize=8)
ax.text(4.75,-1.34,"ranking",ha="center",fontsize=8)
ax.text(6.55,-1.34,"grounding / citation retention",ha="center",fontsize=8)
ax.text(.5,-.04,"(a)",transform=ax.transAxes,ha="center",va="top",fontsize=12)
fig.tight_layout()
fig.savefig(HERE/"figure_10_enterprise_rag_architecture.svg",bbox_inches="tight")
fig.savefig(HERE/"figure_10_enterprise_rag_architecture.png",dpi=300,bbox_inches="tight")
plt.close(fig)
