"""Scientific figure from measured predictions, never from published TuSimple leaderboard values."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser()
p.add_argument("--csv",type=Path,required=True)
p.add_argument("--out",type=Path,required=True)
a=p.parse_args()
with a.csv.open(newline="") as f:
    rows=list(csv.DictReader(f))
if not rows:
    raise RuntimeError("No executed predictions")
ious=np.asarray([float(r["iou"]) for r in rows])
dices=np.asarray([float(r["dice"]) for r in rows])
assert np.all(np.isfinite(ious)) and np.all((ious>=0)&(ious<=1))
a.out.mkdir(parents=True,exist_ok=True)
fig,ax=plt.subplots(figsize=(7.4,4.2),layout="constrained")
idx=np.arange(len(ious))
ax.scatter(idx,ious,label="Mask IoU",s=32,alpha=.85)
ax.scatter(idx,dices,label="Mask Dice",s=32,marker="D",alpha=.75)
ax.axhline(ious.mean(),ls="--",lw=1.5,label=f"Mean frame IoU {ious.mean():.3f}")
ax.set(xlabel="Evaluation frame index",ylabel="Local rasterized-lane overlap",ylim=(0,1.05))
ax.grid(alpha=.16);ax.legend(frameon=False,loc="upper right")
for ext in ("png","pdf","svg"):
    fig.savefig(a.out/f"tusimple_pretrained_unet_measured.{ext}",dpi=220,bbox_inches="tight")
plt.close(fig)
print("Plotted",len(rows),"genuine pretrained-model frame predictions")
