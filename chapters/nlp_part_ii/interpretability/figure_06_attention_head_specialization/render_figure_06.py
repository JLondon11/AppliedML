"""Render NLP Part II Figure 6 from committed aggregate head-score matrices."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
plt.rcParams.update({
 "svg.fonttype":"none","figure.facecolor":"white","axes.facecolor":"white",
 "font.size":9.0,"axes.labelsize":9.2,"xtick.labelsize":7.0,"ytick.labelsize":7.0,
 "axes.linewidth":0.75,"xtick.major.width":0.75,"ytick.major.width":0.75,
})
names=["positional","syntactic","semantic","induction"]
mats=[pd.read_csv(HERE/f"figure_06_{n}_mean_head_scores.csv",index_col=0).to_numpy(float) for n in names]
vmax=max(float(np.nanmax(m)) for m in mats)
fig,axes=plt.subplots(1,4,figsize=(15.4,4.15),constrained_layout=True)
for i,(ax,M) in enumerate(zip(axes,mats)):
 im=ax.imshow(M,origin="lower",aspect="auto",cmap="cividis",vmin=0,vmax=vmax,interpolation="nearest")
 best=np.unravel_index(np.nanargmax(M),M.shape)
 ax.scatter([best[1]],[best[0]],marker="x",s=46,linewidths=1.3,color="#B06D4F")
 ax.set_xlabel("Attention head"); ax.set_ylabel("Layer")
 ax.set_xticks(range(M.shape[1])); ax.set_yticks(range(M.shape[0]))
 ax.tick_params(labelsize=6.5,direction="out",length=3)
 ax.text(.5,-.20,f"({chr(97+i)})",transform=ax.transAxes,ha="center",va="top",fontsize=10.5)
cbar=fig.colorbar(im,ax=axes.ravel().tolist(),fraction=.017,pad=.03,location="right")
cbar.set_label("Mean attention routing score",fontsize=9)
fig.savefig(HERE/"figure_06_attention_head_specialization.svg",bbox_inches="tight")
fig.savefig(HERE/"figure_06_attention_head_specialization.png",dpi=300,bbox_inches="tight")
plt.close(fig)
