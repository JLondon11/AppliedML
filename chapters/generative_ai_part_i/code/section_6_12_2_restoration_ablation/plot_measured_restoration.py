"""Plot genuine original-photograph restoration scores (no invented observations).

Run: python plot_measured_restoration.py /path/to/results/metrics.csv
Produces a paired, condition-level plot instead of a generic four-bar graphic.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS=("noisy","gaussian_sigma_0.8","unconditioned","conditioned")
LABELS=("Corrupted","Gaussian σ=0.8","Unconditioned CNN","Time-conditioned CNN")
COLORS=("#AC7967","#7A8898","#426D92","#227D7B")

def plot(source):
    source=Path(source)
    with source.open(newline="") as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=48: raise ValueError("Exactly 48 recorded original-photo/seed/condition outputs expected")
    seen={(int(r["seed"]),r["image"],float(r["alpha_bar"]),r["model"]) for r in rows}
    if len(seen)!=48: raise ValueError("Duplicate original-photo measurement")
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.8),layout="constrained")
    for ax,(field,ylabel) in zip(axes,(("psnr_db","PSNR (dB)"),("ssim","SSIM"))):
        for j,(arm,label,color) in enumerate(zip(ARMS,LABELS,COLORS)):
            subset=[r for r in rows if r["model"]==arm]
            actual=np.array([float(r[field]) for r in subset])
            if len(actual)!=12 or not np.isfinite(actual).all():
                raise ValueError("Missing/non-finite genuine experimental scores")
            offsets=np.array([-.115,-.065,-.015,.035,.085,.135]*2)
            ax.scatter(j+offsets,actual,s=27,alpha=.65,color=color,
                       edgecolors="white",linewidths=.35)
            ax.hlines(actual.mean(),j-.25,j+.25,color=color,lw=2.8,
                      label=label)
        ax.set(xticks=range(4),xticklabels=LABELS,ylabel=ylabel)
        ax.grid(axis="y",alpha=.17)
        ax.spines[["right","top"]].set_visible(False)
        ax.tick_params(axis="x",labelsize=8)
    axes[0].set_title("A  Original-photo PSNR by method")
    axes[1].set_title("B  Original-photo SSIM by method")
    fig.suptitle("Single-step Gaussian-noise restoration | 2 held-out photographs × 2 noise levels × 3 seeds\nRepeated conditions are not independent image samples",fontsize=10.5)
    out=source.parent/"figures";out.mkdir(exist_ok=True)
    for ext in ("svg","png","pdf"):
        fig.savefig(out/("restoration_scoped_comparison."+ext),dpi=230,bbox_inches="tight")
    plt.close(fig)
    return out

if __name__=="__main__":
    print(plot(sys.argv[1] if len(sys.argv)>1 else Path(__file__).parent/"results"/"metrics.csv"))
