"""Render NLP Part II Figure 4 from committed BERT-Tiny embedding results."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
plt.rcParams.update({
 "svg.fonttype":"none","figure.facecolor":"white","axes.facecolor":"white",
 "font.size":9.0,"axes.labelsize":9.5,"xtick.labelsize":8.0,"ytick.labelsize":8.0,
 "axes.linewidth":0.8,"xtick.major.width":0.8,"ytick.major.width":0.8,
 "legend.fontsize":8.0,"legend.frameon":False,
})
PALETTE={"animals":"#34495E","vehicles":"#71879B","finance":"#4F7068","geography":"#A06F58"}
BANK_FIN="#4F7068"; BANK_GEO="#A06F58"; NEUTRAL="#5B5B5B"
df=pd.read_csv(HERE/"figure_04_bert_tiny_contextual_vectors.csv")
Ddf=pd.read_csv(HERE/"figure_04_centroid_cosine_distances.csv",index_col=0)
meta=pd.read_json(HERE/"figure_04_model_provenance.json",typ="series")
ev=meta["pca_explained_variance_ratio"]
fig,axes=plt.subplots(1,3,figsize=(13.6,4.45))

ax=axes[0]
for cat in ["animals","vehicles","finance","geography"]:
 sub=df[df.category==cat]
 ax.scatter(sub.pc1,sub.pc2,s=42,color=PALETTE[cat],label=cat,edgecolors="white",linewidths=.35)
 offsets={
  "dog":(10,-8),"cat":(8,7),"car":(-22,10),"truck":(10,1),"bus":(8,-11),
  "train":(6,5),"plane":(6,4),"wolf":(6,4),"loan":(6,4),"credit":(6,-2),
  "money":(6,4),"investment":(6,4),"river":(6,4),"shore":(6,4),
  "stream":(6,4),"water":(6,4)
 }
 for r in sub.itertuples(index=False):
  dx,dy=offsets.get(r.target_word,(4,4))
  ax.annotate(r.target_word,(r.pc1,r.pc2),xytext=(dx,dy),textcoords="offset points",fontsize=7.2,color="#333333")
ax.set_xlabel(f"PC 1 ({100*ev[0]:.1f}% variance)"); ax.set_ylabel(f"PC 2 ({100*ev[1]:.1f}% variance)")
ax.legend(loc="best",handletextpad=.45)

ax=axes[1]
bf=df[df.category=="bank_finance"]; bg=df[df.category=="bank_geography"]
ax.scatter(bf.pc1,bf.pc2,s=44,color=BANK_FIN,label="financial bank",edgecolors="white",linewidths=.35)
ax.scatter(bg.pc1,bg.pc2,s=44,color=BANK_GEO,marker="^",label="river bank",edgecolors="white",linewidths=.35)
cf=bf[["pc1","pc2"]].mean().to_numpy(); cg=bg[["pc1","pc2"]].mean().to_numpy()
ax.scatter(*cf,s=88,facecolors="none",edgecolors=BANK_FIN,linewidths=1.4)
ax.scatter(*cg,s=88,facecolors="none",edgecolors=BANK_GEO,linewidths=1.4)
ax.plot([cf[0],cg[0]],[cf[1],cg[1]],color=NEUTRAL,lw=1.1,ls="--")
ax.set_xlabel(f"PC 1 ({100*ev[0]:.1f}% variance)"); ax.set_ylabel(f"PC 2 ({100*ev[1]:.1f}% variance)")
ax.legend(loc="best",handletextpad=.45)

ax=axes[2]
D=Ddf.to_numpy(float); labels=list(Ddf.index)
im=ax.imshow(D,cmap="cividis",aspect="equal",vmin=0,vmax=max(.01,float(np.nanmax(D))))
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels,rotation=48,ha="right",fontsize=7.2)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels,fontsize=7.2)
cb=fig.colorbar(im,ax=ax,fraction=.044,pad=.035); cb.set_label("Cosine distance",fontsize=9)
for i,ax in enumerate(axes):
 ax.grid(False); ax.tick_params(direction="out",length=3)
 ax.text(.5,-.19,f"({chr(97+i)})",transform=ax.transAxes,ha="center",va="top",fontsize=10.5)
fig.tight_layout(w_pad=1.7)
fig.savefig(HERE/"figure_04_embedding_geometry.svg",bbox_inches="tight")
fig.savefig(HERE/"figure_04_embedding_geometry.png",dpi=300,bbox_inches="tight")
plt.close(fig)
