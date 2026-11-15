"""Plot measured COCO SAM segmentation results; does not create or simulate scores."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser()
p.add_argument("--csv", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
args=p.parse_args()
with args.csv.open(newline="") as fh:
    rows=list(csv.DictReader(fh))
if not rows:
    raise SystemExit("No actual prediction measurements to plot")
sam=np.array([float(r["sam_iou"]) for r in rows],dtype=float)
paired=[r for r in rows if r["grabcut_iou"] not in ("",None)]
fig,ax=plt.subplots(figsize=(7.2,4.6),layout="constrained")
ax.hist(sam,bins=np.linspace(0,1,11),edgecolor="white",alpha=.85)
ax.axvline(sam.mean(),linestyle="--",linewidth=1.7,label=f"SAM mean IoU = {sam.mean():.3f}")
ax.set(xlim=(0,1),xlabel="Ground-truth-box prompted SAM mask IoU",ylabel="COCO object count")
ax.legend(loc="upper left")
args.output.mkdir(parents=True,exist_ok=True)
fig.savefig(args.output/"sam_iou_distribution.pdf")
fig.savefig(args.output/"sam_iou_distribution.png",dpi=220)
plt.close(fig)
if paired:
    x=np.array([float(r["grabcut_iou"]) for r in paired])
    y=np.array([float(r["sam_iou"]) for r in paired])
    fig,ax=plt.subplots(figsize=(5,5),layout="constrained")
    ax.scatter(x,y,alpha=.8)
    ax.plot([0,1],[0,1],linestyle="--",linewidth=1,color="gray")
    ax.set(xlim=(0,1),ylim=(0,1),xlabel="GrabCut mask IoU",ylabel="SAM ViT-B mask IoU")
    ax.set_aspect("equal","box")
    fig.savefig(args.output/"sam_grabcut_paired.pdf")
    fig.savefig(args.output/"sam_grabcut_paired.png",dpi=220)
    plt.close(fig)
print(f"Figures generated from {len(rows)} genuine scored objects; {len(paired)} paired comparisons")
