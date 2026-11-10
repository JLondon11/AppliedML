"""Chapter 6 Figure 6.35: paired quality–diversity observations from real NCI experiment.
Run: python figure_6_35_publication.py
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
root=Path(__file__).resolve().parent
with (root/"figure_6_35_measured_source_pass31.csv").open() as fh: rows=list(csv.DictReader(fh))
assert len(rows)==6
by={(int(r["seed"]),r["arm"]):r for r in rows}
seeds=(11,23,37)
blue="#155E75"; orange="#C07851"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.edgecolor":"#B7C3CB"})
fig,ax=plt.subplots(figsize=(8.1,5.0),layout="constrained")
for seed in seeds:
    ran=by[(seed,"random_control")];guided=by[(seed,"surrogate_guided")]
    assert int(ran["candidate_budget"])==int(guided["candidate_budget"])==50
    a=(float(ran["mean_true_descriptor_score"]),int(ran["n_distinct_scaffolds"]))
    b=(float(guided["mean_true_descriptor_score"]),int(guided["n_distinct_scaffolds"]))
    ax.plot([a[0],b[0]],[a[1],b[1]],lw=1.3,color="#9DAEB8",zorder=1)
    ax.annotate("",xy=b,xytext=a,arrowprops=dict(arrowstyle="->",lw=1.3,color="#9DAEB8"))
    ax.scatter(*a,s=75,c=blue,edgecolors="white",linewidth=.8,zorder=3)
    ax.scatter(*b,s=85,c=orange,edgecolors="white",linewidth=.8,zorder=3,marker="D")
    ax.annotate(f"Seed {seed}",a,xytext=(-6,8),textcoords="offset points",fontsize=8,color=blue,ha="right")
for arm,color,marker in (("random_control",blue,"o"),("surrogate_guided",orange,"D")):
    x=np.array([[float(by[(s,arm)]["mean_true_descriptor_score"]),int(by[(s,arm)]["n_distinct_scaffolds"])] for s in seeds])
    mean=x.mean(0)
    ax.scatter(*mean,s=180,c=color,marker=marker,edgecolors="white",linewidth=1.3,zorder=4)
    ax.annotate(f"Mean: {mean[0]:.3f}; {mean[1]:.1f} scaffolds",mean,
                xytext=(9,-15 if arm=="random_control" else 8),textcoords="offset points",
                color=color,fontsize=8.6)
ax.set(xlabel="Mean penalized QED of valid generated molecules (higher is better)",
       ylabel="Distinct Bemis–Murcko scaffolds among 50 valid candidates",
       xlim=(.4,.72),ylim=(7,34))
ax.grid(color="#E8EDF0",linewidth=.8);ax.set_axisbelow(True)
ax.spines[["top","right"]].set_visible(False)
ax.legend(handles=[
    Line2D([0],[0],marker="o",markersize=7,linestyle="None",markerfacecolor=blue,color=blue,label="Random edits"),
    Line2D([0],[0],marker="D",markersize=7,linestyle="None",markerfacecolor=orange,color=orange,label="Surrogate-guided edits")],
    loc="upper center",bbox_to_anchor=(.5,1.11),ncol=2,frameon=False)
for ext in ("pdf","png","svg"):
    fig.savefig(root/("Figure6_35_QualityDiversity."+ext),dpi=220,bbox_inches="tight")
print("Measured six observations plotted; guided proposals NOT compute-matched to random control")
