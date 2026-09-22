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

# Scientific state representation of the executable RAG architecture.
# (a) retrieval score spectrum; (b) reranking decomposition; (c) evidence retention.
fig,axs=plt.subplots(1,3,figsize=(13.8,4.5))

ax=axs[0]
order=rank
labels=[docs[i][0] for i in order]
vals=[float(scores[i]) for i in order]
ax.bar(range(len(order)),vals,width=.68)
ax.set_xticks(range(len(order)),labels)
ax.set_xlabel("Enterprise document")
ax.set_ylabel("TF-IDF cosine similarity")
ax.axhline(vals[2],ls="--",lw=.8)
ax.text(.5,-.20,"(a)",transform=ax.transAxes,ha="center",va="top",fontsize=11)

ax=axs[1]
rr_ids=[i for _,i in rerank]
cos=np.array([scores[i] for i in rr_ids],float)
cov=np.array([len(qt & set(vec.build_analyzer()(texts[i])))/max(1,len(qt)) for i in rr_ids],float)
combined=.75*cos+.25*cov
x=np.arange(len(rr_ids))
ax.plot(x,cos,marker="o",label="Cosine similarity")
ax.plot(x,cov,marker="s",label="Query-token coverage")
ax.plot(x,combined,marker="^",label="Combined rerank score")
ax.set_xticks(x,[docs[i][0] for i in rr_ids])
ax.set_xlabel("Retrieved candidate")
ax.set_ylabel("Score")
ax.set_ylim(0,max(1.0,float(combined.max())*1.12))
ax.legend(frameon=False,fontsize=7)
ax.text(.5,-.20,"(b)",transform=ax.transAxes,ha="center",va="top",fontsize=11)

ax=axs[2]
stage_names=["Retrieved","Reranked","Prompt","Response citation"]
M=np.zeros((len(docs),len(stage_names)))
for r,i in enumerate(rank[:3]): M[i,0]=1
for i in selected: M[i,1]=1; M[i,2]=1
for i in selected:
    if f"[{docs[i][0]}]" in answer: M[i,3]=1
im=ax.imshow(M,cmap="cividis",vmin=0,vmax=1,aspect="auto")
ax.set_yticks(range(len(docs)),[d[0] for d in docs])
ax.set_xticks(range(len(stage_names)),stage_names,rotation=40,ha="right")
ax.set_xlabel("Evidence-retention stage")
ax.set_ylabel("Enterprise document")
cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.04,ticks=[0,1])
cb.set_label("Evidence retained (0/1)")
ax.text(.5,-.20,"(c)",transform=ax.transAxes,ha="center",va="top",fontsize=11)

for ax in axs:
    ax.grid(False); ax.tick_params(direction="out")
fig.tight_layout(w_pad=2.0)
fig.savefig(HERE/"figure_10_enterprise_rag_architecture.svg",bbox_inches="tight")
fig.savefig(HERE/"figure_10_enterprise_rag_architecture.png",dpi=300,bbox_inches="tight")
plt.close(fig)
