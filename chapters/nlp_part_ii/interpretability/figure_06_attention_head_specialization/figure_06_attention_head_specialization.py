"""NLP Part II Figure 6 — Functional Specialization of Transformer Attention Heads.

Frozen inventory:
  Figure 6: Functional Specialization of Transformer Attention Heads
  Section: Interpretability, Mechanistic Understanding, and Internal Representations
  Subsection: Attention Analysis
  Label: fig:attention_interpretability

Scientific design
-----------------
Actual pretrained-model attention weights are extracted from controlled stimuli.
Heads are selected algorithmically using transparent routing scores:

(a) Positional: BERT head maximizing mean attention mass to the immediately
    previous token across non-special tokens.
(b) Syntactic: BERT head maximizing attention from the main verb token to the
    grammatical subject token in a controlled active sentence.
(c) Semantic: BERT head maximizing attention from a target noun to a
    semantically associated noun in a controlled sentence.
(d) Induction-like: GPT-2 head maximizing attention from the second occurrence
    of a repeated token to the token immediately following its first occurrence
    in a repeated-pattern prompt.

These are routing-pattern diagnostics, not causal attributions. The figure does
not claim that attention weights alone explain model behavior.

All selected layer/head IDs, token sequences, score tables, and full attention
matrices are saved as provenance.
"""
from __future__ import annotations
from pathlib import Path
import json, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from huggingface_hub import HfApi

HERE=Path(__file__).resolve().parent
plt.rcParams["svg.fonttype"]="none"

BERT_ID="google-bert/bert-base-uncased"
GPT2_ID="openai-community/gpt2"

def get_word_index(offsets, text, word, occurrence=0):
    low=text.lower()
    starts=[]
    pos=0
    while True:
        i=low.find(word.lower(),pos)
        if i<0: break
        starts.append(i); pos=i+1
    s=starts[occurrence]; e=s+len(word)
    idx=[i for i,(a,b) in enumerate(offsets) if b>a and max(a,s)<min(b,e)]
    if not idx: raise RuntimeError((text,word,occurrence))
    return idx

def mean_block(A, qs, ks):
    return float(A[np.ix_(qs,ks)].mean())

bert_tok=AutoTokenizer.from_pretrained(BERT_ID,use_fast=True)
bert=AutoModel.from_pretrained(BERT_ID,output_attentions=True)
bert.eval()

# Controlled stimuli.
pos_text="The small brown dog quickly crossed the narrow bridge."
syn_text="The scientist carefully analyzed the complex dataset."
sem_text="The physician discussed the diagnosis with the patient."

def bert_attention(text):
    enc=bert_tok(text,return_tensors="pt",return_offsets_mapping=True)
    offsets=enc.pop("offset_mapping")[0].tolist()
    toks=bert_tok.convert_ids_to_tokens(enc["input_ids"][0])
    with torch.no_grad():
        out=bert(**enc,output_attentions=True,return_dict=True)
    # list[layer] -> [heads, seq, seq]
    mats=[x[0].cpu().numpy() for x in out.attentions]
    return enc,toks,offsets,mats

_,pos_toks,pos_off,pos_mats=bert_attention(pos_text)
_,syn_toks,syn_off,syn_mats=bert_attention(syn_text)
_,sem_toks,sem_off,sem_mats=bert_attention(sem_text)

# (a) Positional score: previous-token attention over ordinary tokens.
pos_scores=[]
for l,M in enumerate(pos_mats):
    for h,A in enumerate(M):
        vals=[A[i,i-1] for i in range(2,len(pos_toks)-1)]
        pos_scores.append((float(np.mean(vals)),l,h))
pos_score,pos_l,pos_h=max(pos_scores)
pos_A=pos_mats[pos_l][pos_h]

# (b) Syntactic routing score: main verb "analyzed" -> subject "scientist".
syn_q=get_word_index(syn_off,syn_text,"analyzed")
syn_k=get_word_index(syn_off,syn_text,"scientist")
syn_scores=[]
for l,M in enumerate(syn_mats):
    for h,A in enumerate(M):
        syn_scores.append((mean_block(A,syn_q,syn_k),l,h))
syn_score,syn_l,syn_h=max(syn_scores)
syn_A=syn_mats[syn_l][syn_h]

# (c) Semantic routing score: "diagnosis" -> "patient".
sem_q=get_word_index(sem_off,sem_text,"diagnosis")
sem_k=get_word_index(sem_off,sem_text,"patient")
sem_scores=[]
for l,M in enumerate(sem_mats):
    for h,A in enumerate(M):
        sem_scores.append((mean_block(A,sem_q,sem_k),l,h))
sem_score,sem_l,sem_h=max(sem_scores)
sem_A=sem_mats[sem_l][sem_h]

# (d) Induction-like repeated-pattern routing in GPT-2.
gpt_tok=AutoTokenizer.from_pretrained(GPT2_ID,use_fast=True)
gpt=AutoModelForCausalLM.from_pretrained(GPT2_ID,attn_implementation="eager")
gpt.eval()
prompt="A B C A B"
genc=gpt_tok(prompt,return_tensors="pt")
gtoks=gpt_tok.convert_ids_to_tokens(genc["input_ids"][0])
with torch.no_grad():
    gout=gpt(**genc,output_attentions=True,return_dict=True)
gmats=[x[0].cpu().numpy() for x in gout.attentions]
# Token positions under GPT-2 BPE are verified from exact token IDs.
# Select head maximizing attention from second B token to token after first B
# (the C token), the canonical repeated-pattern induction diagnostic.
Bpos=[i for i,t in enumerate(gtoks) if t.replace("Ġ","")=="B"]
if len(Bpos)!=2: raise RuntimeError(gtoks)
q=Bpos[1]; k=Bpos[0]+1
ind_scores=[]
for l,M in enumerate(gmats):
    for h,A in enumerate(M):
        ind_scores.append((float(A[q,k]),l,h))
ind_score,ind_l,ind_h=max(ind_scores)
ind_A=gmats[ind_l][ind_h]

# Save head-selection score tables.
pd.DataFrame(pos_scores,columns=["score","layer","head"]).sort_values("score",ascending=False).to_csv(HERE/"figure_06_positional_head_scores.csv",index=False)
pd.DataFrame(syn_scores,columns=["score","layer","head"]).sort_values("score",ascending=False).to_csv(HERE/"figure_06_syntactic_head_scores.csv",index=False)
pd.DataFrame(sem_scores,columns=["score","layer","head"]).sort_values("score",ascending=False).to_csv(HERE/"figure_06_semantic_head_scores.csv",index=False)
pd.DataFrame(ind_scores,columns=["score","layer","head"]).sort_values("score",ascending=False).to_csv(HERE/"figure_06_induction_head_scores.csv",index=False)

def save_matrix(name,A,toks):
    df=pd.DataFrame(A,index=toks,columns=toks)
    df.to_csv(HERE/name)

save_matrix("figure_06_positional_attention.csv",pos_A,pos_toks)
save_matrix("figure_06_syntactic_attention.csv",syn_A,syn_toks)
save_matrix("figure_06_semantic_attention.csv",sem_A,sem_toks)
save_matrix("figure_06_induction_attention.csv",ind_A,gtoks)

meta={
 "bert_model_id":BERT_ID,
 "bert_revision":HfApi().model_info(BERT_ID).sha,
 "gpt2_model_id":GPT2_ID,
 "gpt2_revision":HfApi().model_info(GPT2_ID).sha,
 "selected":{
  "positional":{"layer":pos_l,"head":pos_h,"score":pos_score,"stimulus":pos_text},
  "syntactic":{"layer":syn_l,"head":syn_h,"score":syn_score,"stimulus":syn_text,"query":"analyzed","key":"scientist"},
  "semantic":{"layer":sem_l,"head":sem_h,"score":sem_score,"stimulus":sem_text,"query":"diagnosis","key":"patient"},
  "induction_like":{"layer":ind_l,"head":ind_h,"score":ind_score,"stimulus":prompt,"query_position":q,"key_position":k},
 },
 "torch_version":torch.__version__
}
(HERE/"figure_06_model_provenance.json").write_text(json.dumps(meta,indent=2)+"\n")

# Render four true attention matrices. No title/caption/prose in artwork.
fig,axes=plt.subplots(1,4,figsize=(15.6,4.3))
panels=[
 (pos_A,pos_toks,"(a)"),
 (syn_A,syn_toks,"(b)"),
 (sem_A,sem_toks,"(c)"),
 (ind_A,gtoks,"(d)")
]
for ax,(A,toks,lab) in zip(axes,panels):
    im=ax.imshow(A,cmap="cividis",vmin=0,vmax=max(.35,float(np.quantile(A,.995))),aspect="auto")
    ax.set_xticks(range(len(toks))); ax.set_xticklabels(toks,rotation=90,fontsize=6)
    ax.set_yticks(range(len(toks))); ax.set_yticklabels(toks,fontsize=6)
    ax.set_xlabel("Key token"); ax.set_ylabel("Query token")
    ax.text(.5,-.26,lab,transform=ax.transAxes,ha="center",va="top",fontsize=12)
    ax.tick_params(length=2,width=.6,direction="out")
fig.tight_layout(w_pad=1.3)
cbar=fig.colorbar(im,ax=axes.ravel().tolist(),fraction=.018,pad=.02)
cbar.set_label("Attention weight")
fig.savefig(HERE/"figure_06_attention_head_specialization.svg",bbox_inches="tight")
fig.savefig(HERE/"figure_06_attention_head_specialization.png",dpi=300,bbox_inches="tight")
plt.close(fig)
