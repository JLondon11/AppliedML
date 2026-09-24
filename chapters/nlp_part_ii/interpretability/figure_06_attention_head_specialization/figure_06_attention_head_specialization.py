"""NLP Part II Figure 6 — Functional Specialization of Transformer Attention Heads.

Scientific remediation: specialization is measured across multiple controlled
stimuli per diagnostic rather than selecting a head from one sentence. Panels
show layer-by-head mean routing scores computed from actual pretrained-model
attention weights. These are descriptive routing diagnostics, not causal
attributions of model behavior.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from huggingface_hub import HfApi

HERE=Path(__file__).resolve().parent
plt.rcParams["svg.fonttype"]="none"
BERT_ID="google-bert/bert-base-uncased"
GPT2_ID="openai-community/gpt2"

POS_TEXTS=[
 "The small brown dog quickly crossed the narrow bridge.",
 "A careful engineer quietly inspected the new machine.",
 "The young student slowly opened the heavy textbook.",
 "Several bright birds suddenly left the old tree.",
 "The research team carefully measured the final sample.",
]
SYN_STIMULI=[
 ("The scientist carefully analyzed the complex dataset.","analyzed","scientist"),
 ("The engineer thoroughly tested the prototype.","tested","engineer"),
 ("The physician carefully examined the patient.","examined","physician"),
 ("The lawyer closely reviewed the contract.","reviewed","lawyer"),
 ("The teacher patiently graded the assignment.","graded","teacher"),
]
SEM_STIMULI=[
 ("The physician discussed the diagnosis with the patient.","diagnosis","patient"),
 ("The mechanic repaired the engine inside the vehicle.","engine","vehicle"),
 ("The banker approved the loan for the customer.","loan","customer"),
 ("The biologist observed the species in the habitat.","species","habitat"),
 ("The programmer fixed the bug in the software.","bug","software"),
]
IND_STIMULI=[
 ("A B C A B","B"),
 ("red blue green red blue","blue"),
 ("cat dog bird cat dog","dog"),
 ("one two three one two","two"),
 ("alpha beta gamma alpha beta","beta"),
]

def word_indices(offsets,text,word,occurrence=0):
    low=text.lower(); starts=[]; p=0
    while True:
        i=low.find(word.lower(),p)
        if i<0: break
        starts.append(i); p=i+1
    if occurrence>=len(starts): raise RuntimeError((text,word,occurrence))
    s=starts[occurrence]; e=s+len(word)
    idx=[i for i,(a,b) in enumerate(offsets) if b>a and max(a,s)<min(b,e)]
    if not idx: raise RuntimeError((text,word))
    return idx

def mean_block(A,qs,ks):
    return float(A[np.ix_(qs,ks)].mean())

bert_tok=AutoTokenizer.from_pretrained(BERT_ID,use_fast=True)
bert=AutoModel.from_pretrained(BERT_ID,output_attentions=True)
bert.eval()

def bert_attention(text):
    enc=bert_tok(text,return_tensors="pt",return_offsets_mapping=True)
    offsets=enc.pop("offset_mapping")[0].tolist()
    with torch.no_grad():
        out=bert(**enc,output_attentions=True,return_dict=True)
    mats=np.stack([x[0].cpu().numpy() for x in out.attentions]) # L,H,Q,K
    return offsets,mats

pos_acc=[]; syn_acc=[]; sem_acc=[]
for text in POS_TEXTS:
    _,M=bert_attention(text)
    S=np.zeros(M.shape[:2],float)
    for l in range(M.shape[0]):
        for h in range(M.shape[1]):
            A=M[l,h]
            vals=[A[i,i-1] for i in range(2,A.shape[0]-1)]
            S[l,h]=float(np.mean(vals))
    pos_acc.append(S)

for text,qword,kword in SYN_STIMULI:
    offsets,M=bert_attention(text)
    q=word_indices(offsets,text,qword); k=word_indices(offsets,text,kword)
    S=np.zeros(M.shape[:2],float)
    for l in range(M.shape[0]):
        for h in range(M.shape[1]): S[l,h]=mean_block(M[l,h],q,k)
    syn_acc.append(S)

for text,qword,kword in SEM_STIMULI:
    offsets,M=bert_attention(text)
    q=word_indices(offsets,text,qword); k=word_indices(offsets,text,kword)
    S=np.zeros(M.shape[:2],float)
    for l in range(M.shape[0]):
        for h in range(M.shape[1]): S[l,h]=mean_block(M[l,h],q,k)
    sem_acc.append(S)

POS=np.mean(np.stack(pos_acc),axis=0)
SYN=np.mean(np.stack(syn_acc),axis=0)
SEM=np.mean(np.stack(sem_acc),axis=0)

gpt_tok=AutoTokenizer.from_pretrained(GPT2_ID,use_fast=True)
gpt=AutoModelForCausalLM.from_pretrained(GPT2_ID,attn_implementation="eager")
gpt.eval()
ind_acc=[]
for prompt,repeated in IND_STIMULI:
    enc=gpt_tok(prompt,return_tensors="pt")
    toks=gpt_tok.convert_ids_to_tokens(enc["input_ids"][0])
    norm=[t.replace("Ġ","").lower() for t in toks]
    pos=[i for i,t in enumerate(norm) if t==repeated.lower()]
    if len(pos)!=2: raise RuntimeError((prompt,toks,repeated))
    q=pos[1]; k=pos[0]+1
    with torch.no_grad():
        out=gpt(**enc,output_attentions=True,return_dict=True)
    M=np.stack([x[0].cpu().numpy() for x in out.attentions])
    S=np.zeros(M.shape[:2],float)
    for l in range(M.shape[0]):
        for h in range(M.shape[1]): S[l,h]=float(M[l,h,q,k])
    ind_acc.append(S)
IND=np.mean(np.stack(ind_acc),axis=0)

for name,M in [("positional",POS),("syntactic",SYN),("semantic",SEM),("induction",IND)]:
    pd.DataFrame(M,index=np.arange(M.shape[0]),columns=np.arange(M.shape[1])).to_csv(
        HERE/f"figure_06_{name}_mean_head_scores.csv",index_label="layer")
    flat=[(float(M[l,h]),l,h) for l in range(M.shape[0]) for h in range(M.shape[1])]
    pd.DataFrame(flat,columns=["mean_score","layer","head"]).sort_values(
        "mean_score",ascending=False).to_csv(HERE/f"figure_06_{name}_head_ranking.csv",index=False)

meta={
 "bert_model_id":BERT_ID,
 "bert_revision":HfApi().model_info(BERT_ID).sha,
 "gpt2_model_id":GPT2_ID,
 "gpt2_revision":HfApi().model_info(GPT2_ID).sha,
 "num_positional_stimuli":len(POS_TEXTS),
 "num_syntactic_stimuli":len(SYN_STIMULI),
 "num_semantic_stimuli":len(SEM_STIMULI),
 "num_induction_stimuli":len(IND_STIMULI),
 "interpretation":"Mean routing scores across controlled stimuli; descriptive attention diagnostics, not causal attribution.",
 "torch_version":torch.__version__,
}
(HERE/"figure_06_model_provenance.json").write_text(json.dumps(meta,indent=2)+"\n")

mats=[POS,SYN,SEM,IND]
vmax=max(float(m.max()) for m in mats)
fig,axes=plt.subplots(1,4,figsize=(15.8,4.35),constrained_layout=True)
for i,(ax,M) in enumerate(zip(axes,mats)):
    im=ax.imshow(M,origin="lower",aspect="auto",cmap="cividis",vmin=0,vmax=vmax)
    best=np.unravel_index(np.argmax(M),M.shape)
    ax.scatter([best[1]],[best[0]],marker="x",s=38,linewidths=1.1)
    ax.set_xlabel("Attention head")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(M.shape[1]))
    ax.set_yticks(range(M.shape[0]))
    ax.tick_params(labelsize=6,direction="out")
    ax.text(.5,-.20,f"({chr(97+i)})",transform=ax.transAxes,ha="center",va="top",fontsize=11)
cbar=fig.colorbar(im,ax=axes.ravel().tolist(),fraction=.018,pad=.035,location="right")
cbar.set_label("Mean attention routing score")
fig.tight_layout(w_pad=1.5)
fig.savefig(HERE/"figure_06_attention_head_specialization.svg",bbox_inches="tight")
fig.savefig(HERE/"figure_06_attention_head_specialization.png",dpi=300,bbox_inches="tight")
plt.close(fig)
