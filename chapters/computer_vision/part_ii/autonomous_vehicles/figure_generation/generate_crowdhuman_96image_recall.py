"""Recreate Figure 5.50 from the independently executed 96-image CrowdHuman study.
Recorded source: https://github.com/JLondon11/AppliedML/actions/runs/42323774125
Only actual aggregated model-inference counts are plotted.
"""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
here=Path(__file__).resolve().parent
with (here/"crowdhuman_96image_counts.csv").open(newline="") as fh:
    r=list(csv.DictReader(fh))
rows=[(v["stratum"],int(v["matched"]),int(v["total"])) for v in r]
assert rows==[("Low",653,861),("Medium",325,638),("Heavy",92,797)]
values=[good/total for _,good,total in rows]
fig,ax=plt.subplots(figsize=(7.2,4.35),layout="constrained")
bars=ax.bar([k for k,_,_ in rows],values,width=.58,
            color=["#167994","#9c7b56","#bd615a"])
for bar,(_,good,total),v in zip(bars,rows,values):
    ax.text(bar.get_x()+bar.get_width()/2,v+.025,
            f"{v:.3f} ({good}/{total})",ha="center",fontsize=10)
ax.set(xlabel="Occlusion proxy category",
       ylabel="Full-body detection recall at IoU ≥ 0.50",
       ylim=(0,1.08))
ax.grid(axis="y",alpha=.18)
ax.set_axisbelow(True)
for ext in ("pdf","png","svg"):
    fig.savefig(here/f"crowdhuman_96image_recall.{ext}",dpi=240,bbox_inches="tight")
plt.close(fig)
