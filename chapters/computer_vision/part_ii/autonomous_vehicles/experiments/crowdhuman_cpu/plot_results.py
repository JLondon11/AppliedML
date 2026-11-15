"""Plot per-person metrics derived from genuine CrowdHuman detector predictions."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument("--csv", type=Path,required=True)
p.add_argument("--out",type=Path,required=True)
a=p.parse_args()
with a.csv.open(newline="") as f:
    rows=list(csv.DictReader(f))
if len(rows)!=3:
    raise ValueError("Expected exactly three occlusion strata")
names, values, n=[],[],[]
for r in rows:
    names.append(r["stratum"])
    n.append(int(r["total"]))
    values.append(float(r["local_recall_IoU50"]) if r["local_recall_IoU50"] else None)
if not sum(n):
    raise ValueError("No real person annotations scored")
fig,ax=plt.subplots(figsize=(6.5,4.1),layout="constrained")
bars=ax.bar(names,[v if v is not None else 0 for v in values],color=["#3684a7","#688f71","#b16d5a"])
for bar,count,v in zip(bars,n,values):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.025,
            f"{v:.3f} (n={count})" if v is not None else "No instances",
            ha="center",fontsize=9)
ax.set(ylim=(0,1.16),ylabel="Local full-body recall at IoU ≥ 0.50",
       xlabel="Visible/full-body box area proxy stratum")
ax.grid(axis="y",alpha=.15)
a.out.mkdir(parents=True,exist_ok=True)
for ext in ("pdf","png","svg"):
    fig.savefig(a.out/f"crowdhuman_occlusion_recall.{ext}",dpi=220,bbox_inches="tight")
print(f"Scored {sum(n)} actual CrowdHuman reference people")
