"""NLP Part II Figure 7 — BERT sentiment computation as measured tensor states.

This replaces an arrow-chain architecture diagram with a scientific rendering
computed from an actual pretrained BERT forward pass. The panels show:
(a) token-by-hidden-state structure after the final encoder layer,
(b) layer-wise [CLS] state evolution, and
(c) classifier logits/probabilities from the sequence-classification head.

No benchmark performance is implied by this conceptual forward-pass rendering.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import HfApi

HERE=Path(__file__).resolve().parent
MODEL="google-bert/bert-base-uncased"
SAMPLE="This film is beautifully acted and emotionally compelling."

tok=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForSequenceClassification.from_pretrained(MODEL,num_labels=2)
model.eval()
enc=tok(SAMPLE,return_tensors="pt",truncation=True,max_length=256)
with torch.no_grad():
    out=model(**enc,output_hidden_states=True,return_dict=True)

tokens=tok.convert_ids_to_tokens(enc["input_ids"][0])
H=np.stack([h[0].cpu().numpy() for h in out.hidden_states])  # (emb+layers, tokens, hidden)
last=H[-1]
cls=H[:,0,:]
norm=np.linalg.norm(cls,axis=1)
base=cls[0]
cos=np.array([np.dot(v,base)/(np.linalg.norm(v)*np.linalg.norm(base)+1e-12) for v in cls])
logits=out.logits[0].cpu().numpy()
prob=torch.softmax(out.logits[0],dim=0).cpu().numpy()

# Save numerical states used by the figure.
pd.DataFrame(last,index=tokens).to_csv(HERE/"figure_07_final_hidden_state.csv",index_label="token")
pd.DataFrame({"layer":np.arange(len(norm)),"cls_norm":norm,"cosine_to_embedding_cls":cos}).to_csv(
    HERE/"figure_07_cls_layer_diagnostics.csv",index=False)
pd.DataFrame({"class":["negative","positive"],"logit":logits,"probability":prob}).to_csv(
    HERE/"figure_07_classifier_output.csv",index=False)

meta={
    "model_id":MODEL,
    "model_revision":HfApi().model_info(MODEL).sha,
    "sample_review":SAMPLE,
    "token_count":int(enc["input_ids"].shape[1]),
    "hidden_size":int(model.config.hidden_size),
    "encoder_layers":int(model.config.num_hidden_layers),
    "num_labels":2,
    "rendering":"measured tensor/state diagnostics from one pretrained-model forward pass",
    "caveat":"Conceptual computation rendering only; no task-performance claim.",
    "torch_version":torch.__version__,
}
(HERE/"figure_07_model_provenance.json").write_text(json.dumps(meta,indent=2)+"\n")

fig,axs=plt.subplots(1,3,figsize=(12.6,4.1))

# (a) Final hidden-state matrix, standardized within each hidden dimension for legibility.
M=last[:,:96]
mu=M.mean(axis=0,keepdims=True); sd=M.std(axis=0,keepdims=True)+1e-8
Z=(M-mu)/sd
im=axs[0].imshow(Z,aspect="auto",cmap="coolwarm",vmin=-2.5,vmax=2.5)
axs[0].set_xlabel("Hidden dimension (first 96)")
axs[0].set_ylabel("Input token")
axs[0].set_yticks(range(len(tokens)))
axs[0].set_yticklabels(tokens,fontsize=7)
cb=fig.colorbar(im,ax=axs[0],fraction=.046,pad=.04)
cb.set_label("Standardized activation")

# (b) Layer-wise [CLS] representation evolution.
layers=np.arange(len(norm))
axs[1].plot(layers,norm,marker="o",ms=3,label=r"$\|h_{CLS}\|_2$")
ax2=axs[1].twinx()
ax2.plot(layers,cos,marker="s",ms=3,linestyle="--",label="cosine to embedding [CLS]")
axs[1].set_xlabel("Embedding / encoder layer index")
axs[1].set_ylabel("[CLS] state norm")
ax2.set_ylabel("Cosine similarity")
axs[1].set_xticks(layers[::max(1,len(layers)//6)])
lines=axs[1].get_lines()+ax2.get_lines()
axs[1].legend(lines,[l.get_label() for l in lines],frameon=False,fontsize=7,loc="best")

# (c) Actual classification-head state.
x=np.arange(2)
axs[2].bar(x,prob,width=.58)
axs[2].set_xticks(x,["Negative","Positive"])
axs[2].set_ylabel("Softmax probability")
axs[2].set_ylim(0,1)
for i,(p,l) in enumerate(zip(prob,logits)):
    axs[2].text(i,p+.025,f"p={p:.3f}\nlogit={l:.2f}",ha="center",va="bottom",fontsize=8)

for i,ax in enumerate(axs):
    ax.grid(False)
    ax.tick_params(direction="out")
    ax.text(.5,-.19,f"({chr(97+i)})",transform=ax.transAxes,ha="center",va="top",fontsize=11)

fig.tight_layout(w_pad=2.0)
fig.savefig(HERE/"figure_07_bert_sentiment_architecture.svg",bbox_inches="tight")
fig.savefig(HERE/"figure_07_bert_sentiment_architecture.png",dpi=300,bbox_inches="tight")
plt.close(fig)
