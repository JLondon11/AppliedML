"""Render NLP Part II Figure 12 from committed claim-level SciFact results.

This renderer is intentionally separate from the expensive experiment. The authoritative
claim-level results and bootstrap metrics are produced by
figure_12_hallucination_reduction_rag.py. This file only renders those stored results.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
plt.rcParams.update({
    "svg.fonttype":"none","figure.facecolor":"white","axes.facecolor":"white",
    "font.size":9.0,"axes.labelsize":9.5,"xtick.labelsize":8.0,"ytick.labelsize":8.0,
    "axes.linewidth":0.8,"xtick.major.width":0.8,"ytick.major.width":0.8,
})
df=pd.read_csv(HERE/"figure_12_claim_level_results.csv")
metrics=json.loads((HERE/"figure_12_metrics.json").read_text())

x0=df["standalone_unsupported_rate"].to_numpy(float)
x1=df["rag_unsupported_rate"].to_numpy(float)
dmean=float(metrics["mean_paired_change_rag_minus_standalone"])
dci=np.asarray(metrics["paired_change_bootstrap_95ci"],float)
miss_rate=float(metrics["rag_retrieval_miss_rate"])
genfail_rate=float(metrics["rag_generation_failure_given_hit_population_rate"])

fig,axs=plt.subplots(1,2,figsize=(10.2,4.45))
ax=axs[0]
ax.scatter(x0,x1,s=22,alpha=.58,color="#5A7D7C",edgecolors="none")
ax.plot([0,1],[0,1],ls="--",lw=1.1,color="#7A7A7A")
ax.set_xlabel("Standalone unsupported-term fraction")
ax.set_ylabel("Retrieval-grounded unsupported-term fraction")
ax.set_xlim(-.02,1.02); ax.set_ylim(-.02,1.02)
ax.text(.04,.96,
        f"mean Δ (RAG − standalone) = {dmean:.3f}\n95% bootstrap CI [{dci[0]:.3f}, {dci[1]:.3f}]",
        transform=ax.transAxes,ha="left",va="top",fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.25",fc="#FAFAF8",ec="#B5B0AA",alpha=.96))

ax=axs[1]
cats=["Retrieval miss","Generation failure\nafter retrieval hit"]
vv=[miss_rate,genfail_rate]
bars=ax.bar([0,1],vv,width=.54,color=["#43566B","#B06D4F"],edgecolor="none")
ax.set_xticks([0,1],cats)
ax.set_ylabel("Fraction of evaluated claims")
ax.set_ylim(0,1)
for bar,v in zip(bars,vv):
    ax.text(bar.get_x()+bar.get_width()/2,v+.025,f"{v:.3f}",ha="center",va="bottom",fontsize=8)

for i,ax in enumerate(axs):
    ax.tick_params(direction="out")
    ax.grid(False)
    ax.text(.5,-.23,f"({chr(97+i)})",transform=ax.transAxes,ha="center",va="top",fontsize=11)

fig.tight_layout(w_pad=2.3)
fig.savefig(HERE/"figure_12_hallucination_reduction_rag.svg",bbox_inches="tight")
fig.savefig(HERE/"figure_12_hallucination_reduction_rag.png",dpi=300,bbox_inches="tight")
plt.close(fig)
