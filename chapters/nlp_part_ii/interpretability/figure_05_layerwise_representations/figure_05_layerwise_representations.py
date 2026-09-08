"""NLP Part II Figure 5 — Layer-Wise Evolution of Transformer Representations.

Frozen inventory:
  Figure 5: Layer-Wise Evolution of Transformer Representations
  Section: Interpretability, Mechanistic Understanding, and Internal Representations
  Subsection: Internal Representations in Transformer Architectures

This experiment runs bert-base-uncased and derives all plotted values from
actual hidden states. No layer is assigned a single function. The panels
measure gradual representational change using:
(a) centered-kernel alignment (linear CKA) between layer representations;
(b) lexical identity retention: cosine similarity of the same target word
    across distinct sentence contexts, layer by layer;
(c) contextual relational refinement: separation between financial and
    geographic senses of the polysemous token "bank", relative to within-sense
    dispersion.

All hidden-state summary values and model revision are saved as provenance.
"""
from pathlib import Path
import json, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import HfApi

MODEL_ID="google-bert/bert-base-uncased"
HERE=Path(__file__).resolve().parent
plt.rcParams["svg.fonttype"]="none"

sentences=[
("dog","The dog chased a ball across the field."),
("dog","A dog waited patiently beside the front door."),
("dog","Their dog slept beneath the kitchen table."),
("cat","The cat watched birds from the window."),
("cat","A cat crossed the quiet garden at dusk."),
("cat","Her cat slept on the warm blanket."),
("river","The river flowed through the broad valley."),
("river","A river crossed the plain below the mountains."),
("river","The river carried cold water toward the sea."),
("loan","The bank approved the business loan yesterday."),
("loan","The lender offered a loan with a lower rate."),
("loan","They repaid the loan before the final deadline."),
("bank_finance","The bank approved the business loan yesterday."),
("bank_finance","She deposited money at the bank before work."),
("bank_finance","The bank offered customers a lower interest rate."),
("bank_finance","The bank reviewed the company's credit application."),
("bank_finance","Investors met with the bank to discuss financing."),
("bank_geography","They rested on the bank of the river."),
("bank_geography","Trees grew along the muddy river bank."),
("bank_geography","The canoe reached the bank before the storm."),
("bank_geography","Wildflowers covered the steep bank beside the stream."),
("bank_geography","The water rose above the grassy bank after the rain."),
]
targets={"dog":"dog","cat":"cat","river":"river","loan":"loan",
         "bank_finance":"bank","bank_geography":"bank"}

def target_indices(offsets,sentence,target):
    s=sentence.lower().index(target); e=s+len(target); out=[]
    for i,(a,b) in enumerate(offsets):
        if b>a and max(a,s)<min(b,e): out.append(i)
    if not out: raise RuntimeError((sentence,target))
    return out

def linear_cka(X,Y):
    X=X-X.mean(0,keepdims=True); Y=Y-Y.mean(0,keepdims=True)
    hs=np.linalg.norm(X.T@Y,"fro")**2
    den=np.linalg.norm(X.T@X,"fro")*np.linalg.norm(Y.T@Y,"fro")
    return float(hs/den)

tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModel.from_pretrained(MODEL_ID)
model.eval()
all_rows=[]; matrices=None
with torch.no_grad():
    for sid,(group,sentence) in enumerate(sentences):
        enc=tok(sentence,return_tensors="pt",return_offsets_mapping=True,truncation=True,max_length=96)
        offsets=enc.pop("offset_mapping")[0].tolist()
        idx=target_indices(offsets,sentence,targets[group])
        out=model(**enc,output_hidden_states=True,return_dict=True)
        if matrices is None: matrices=[[] for _ in out.hidden_states]
        for layer,h in enumerate(out.hidden_states):
            v=h[0,idx,:].mean(0).cpu().numpy()
            matrices[layer].append(v)
            row={"sentence_id":sid,"group":group,"target":targets[group],"sentence":sentence,"layer":layer}
            row.update({f"h{i:03d}":float(x) for i,x in enumerate(v)})
            all_rows.append(row)
df=pd.DataFrame(all_rows)
df.to_csv(HERE/"figure_05_hidden_vectors.csv",index=False)

mats=[np.stack(x) for x in matrices]; L=len(mats)
cka=np.zeros((L,L))
for i in range(L):
    for j in range(L): cka[i,j]=linear_cka(mats[i],mats[j])
pd.DataFrame(cka,index=range(L),columns=range(L)).to_csv(HERE/"figure_05_layer_cka.csv")

def cos(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
lex=[]
for layer in range(L):
    vals=[]
    for group in ["dog","cat","river","loan"]:
        X=mats[layer][df[df.layer==layer].reset_index().query("group==@group").index] if False else None
        sub=df[(df.layer==layer)&(df.group==group)]
        H=sub[[c for c in df.columns if len(c)==4 and c[0]=="h" and c[1:].isdigit()]].to_numpy()
        vals += [cos(H[i],H[j]) for i in range(len(H)) for j in range(i+1,len(H))]
    lex.append(np.mean(vals))
pd.DataFrame({"layer":range(L),"same_word_cross_context_cosine":lex}).to_csv(HERE/"figure_05_lexical_retention.csv",index=False)

sep=[]
for layer in range(L):
    sub=df[df.layer==layer]; hc=[c for c in df.columns if len(c)==4 and c[0]=="h" and c[1:].isdigit()]
    F=sub[sub.group=="bank_finance"][hc].to_numpy(); G=sub[sub.group=="bank_geography"][hc].to_numpy()
    cf,cg=F.mean(0),G.mean(0)
    between=1-cos(cf,cg)
    within=np.mean([1-cos(x,cf) for x in F]+[1-cos(x,cg) for x in G])
    sep.append(between/(within+1e-12))
pd.DataFrame({"layer":range(L),"bank_sense_separation_ratio":sep}).to_csv(HERE/"figure_05_contextual_separation.csv",index=False)

meta={"model_id":MODEL_ID,"model_revision":HfApi().model_info(MODEL_ID).sha,
      "hidden_size":int(mats[0].shape[1]),"layers_including_embedding":L,
      "num_sentences":len(sentences),"torch_version":torch.__version__}
(HERE/"figure_05_model_provenance.json").write_text(json.dumps(meta,indent=2)+"\n")

fig,ax=plt.subplots(1,3,figsize=(13.7,4.4))
im=ax[0].imshow(cka,origin="lower",aspect="auto",cmap="cividis",vmin=0,vmax=1)
ax[0].set(xlabel="Layer",ylabel="Layer"); fig.colorbar(im,ax=ax[0],fraction=.046,pad=.04,label="Linear CKA")
ax[1].plot(range(L),lex,marker="o",ms=3,lw=1.5); ax[1].set(xlabel="Layer",ylabel="Same-word cross-context cosine")
ax[2].plot(range(L),sep,marker="o",ms=3,lw=1.5); ax[2].set(xlabel="Layer",ylabel="Bank sense separation / within-sense dispersion")
for lab,a in zip(["(a)","(b)","(c)"],ax):
    a.text(.5,-.20,lab,transform=a.transAxes,ha="center",va="top",fontsize=12)
    a.tick_params(direction="out",width=.8); a.grid(False)
fig.tight_layout(w_pad=1.8)
fig.savefig(HERE/"figure_05_layerwise_representations.svg",bbox_inches="tight")
fig.savefig(HERE/"figure_05_layerwise_representations.png",dpi=300,bbox_inches="tight")
plt.close(fig)
