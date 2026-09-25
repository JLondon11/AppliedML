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
df=pd.read_csv(HERE/"figure_12_claim_level_results.csv")
metrics=json.loads((HERE/"figure_12_metrics.json").read_text())

x0=df["standalone_unsupported_rate"].to_numpy(float)
x1=df["rag_unsupported_rate"].to_numpy(float)
dmean=float(metrics["mean_paired_change_rag_minus_standalone"])
dci=np.asarray(metrics["paired_change_bootstrap_95ci"],float)
miss_rate=float(metrics["rag_retrieval_miss_rate"])
genfail_rate=float(metrics["rag_generation_failure_given_hit_population_rate"])

fig,axs=plt.subplots(1,2,figsize=(10.4,4.6))
ax=axs[0]
ax.scatter(x0,x1,s=18,alpha=.55,color="#5A7D7C",edgecolors="none")
ax.plot([0,1],[0,1],ls="--",lw=1.0,color="#666666")
ax.set_xlabel("Standalone unsupported-term fraction")
ax.set_ylabel("Retrieval-grounded unsupported-term fraction")
ax.set_xlim(-.02,1.02); ax.set_ylim(-.02,1.02)
ax.text(.04,.96,
        f"mean Δ (RAG − standalone) = {dmean:.3f}\n95% bootstrap CI [{dci[0]:.3f}, {dci[1]:.3f}]",
        transform=ax.transAxes,ha="left",va="top",fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25",fc="white",ec="#999999",alpha=.88))

ax=axs[1]
cats=["Retrieval miss","Generation failure\nafter retrieval hit"]
vv=[miss_rate,genfail_rate]
bars=ax.bar([0,1],vv,width=.58,color=["#43566B","#B26E3B"])
ax.set_xticks([0,1],cats)
ax.set_ylabel("Fraction of evaluated claims")
ax.set_ylim(0,1)
for bar,v in zip(bars,vv):
    ax.text(bar.get_x()+bar.get_width()/2,v+.025,f"{v:.3f}",ha="center",va="bottom",fontsize=8)

for i,ax in enumerate(axs):
    ax.tick_params(direction="out")
    ax.grid(False)
    ax.text(.5,-.23,f"({chr(97+i)})",transform=ax.transAxes,ha="center",va="top",fontsize=11)

fig.tight_layout(w_pad=2.6)
fig.savefig(HERE/"figure_12_hallucination_reduction_rag.svg",bbox_inches="tight")
fig.savefig(HERE/"figure_12_hallucination_reduction_rag.png",dpi=300,bbox_inches="tight")
plt.close(fig)
