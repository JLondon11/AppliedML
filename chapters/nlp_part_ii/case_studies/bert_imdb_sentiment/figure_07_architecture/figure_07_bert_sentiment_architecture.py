from pathlib import Path
import json, torch, matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import HfApi
HERE=Path(__file__).resolve().parent; MODEL="google-bert/bert-base-uncased"
tok=AutoTokenizer.from_pretrained(MODEL); model=AutoModelForSequenceClassification.from_pretrained(MODEL,num_labels=2); model.eval()
sample="This film is beautifully acted and emotionally compelling."
enc=tok(sample,return_tensors="pt",truncation=True,max_length=256)
with torch.no_grad(): out=model(**enc,output_hidden_states=True,return_dict=True)
seq=int(enc["input_ids"].shape[1]); H=int(model.config.hidden_size); L=int(model.config.num_hidden_layers); V=int(model.config.vocab_size)
P=sum(p.numel() for p in model.parameters()); clf=sum(p.numel() for p in model.classifier.parameters())
meta={"model_id":MODEL,"model_revision":HfApi().model_info(MODEL).sha,"sample_review":sample,"token_count":seq,"vocab_size":V,"hidden_size":H,"encoder_layers":L,"total_parameters":P,"classification_head_parameters":clf,"pretrained_backbone_parameters":P-clf,"num_labels":2,"torch_version":torch.__version__}
(HERE/"figure_07_model_provenance.json").write_text(json.dumps(meta,indent=2)+"\n")
fig,ax=plt.subplots(figsize=(13.5,4)); ax.set_xlim(-.5,6.5); ax.set_ylim(-1.25,1.35); ax.axis("off")
labels=[("Review text","UTF-8 string"),("WordPiece IDs","1 x "+str(seq)),("Token + position\nembeddings",str(seq)+" x "+str(H)),("BERT encoder\nL="+str(L)+" layers",str(seq)+" x "+str(H)),("[CLS] state","1 x "+str(H)),("Linear classifier",str(H)+" -> 2"),("Softmax","p(y=0), p(y=1)")]
for i,(name,shape) in enumerate(labels):
 ax.plot([i,i],[0,.62],lw=7,solid_capstyle="butt",alpha=.16); ax.text(i,.72,name,ha="center",va="bottom",fontsize=9); ax.text(i,-.12,shape,ha="center",va="top",fontsize=8)
 if i<6: ax.annotate("",xy=(i+.88,.30),xytext=(i+.12,.30),arrowprops=dict(arrowstyle="->",lw=1.15))
ax.axvline(4.48,ymin=.16,ymax=.83,ls="--",lw=.9); ax.text(2.8,-.72,"pretrained representation reused",ha="center",fontsize=8); ax.text(5.35,-.72,"supervised task adaptation",ha="center",fontsize=8)
ax.annotate("",xy=(4.42,-.58),xytext=(1.05,-.58),arrowprops=dict(arrowstyle="<->",lw=.8)); ax.annotate("",xy=(5.95,-.58),xytext=(4.55,-.58),arrowprops=dict(arrowstyle="<->",lw=.8))
ax.text(.5,-.23,"(a)",transform=ax.transAxes,ha="center",va="top",fontsize=12); fig.tight_layout()
fig.savefig(HERE/"figure_07_bert_sentiment_architecture.svg",bbox_inches="tight"); fig.savefig(HERE/"figure_07_bert_sentiment_architecture.png",dpi=300,bbox_inches="tight"); plt.close(fig)
