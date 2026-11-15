"""Publication-ready paired per-image failure plots from executed real predictions.

Inputs are original experiment 'per_image.csv' derived from running both real
torchvision pretrained models on all original PennFudanPed images.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--per-image",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    with a.per_image.open(newline="") as f:
        rows=list(csv.DictReader(f))
    groups={name:{r["image"]:r for r in rows if r["model"]==name}
            for name in ("MobileNetV3-FPN","ResNet50-FPN")}
    mobile,resnet=groups.values()
    if len(mobile)!=170 or set(mobile)!=set(resnet):
        raise ValueError("Expect 170 exact paired original photographs")
    names=sorted(mobile)
    gt=np.asarray([int(mobile[n]["gt"]) for n in names])
    m=np.asarray([int(mobile[n]["matched"]) for n in names])
    r=np.asarray([int(resnet[n]["matched"]) for n in names])
    differences=[n for n in names if int(mobile[n]["matched"])!=int(resnet[n]["matched"])]
    if sum(r-m)!=6 or len(differences)!=4 or (gt!=np.array([int(resnet[n]["gt"]) for n in names])).any():
        raise ValueError("Raw per-image results do not match independently verified full-set evidence")
    fig,axes=plt.subplots(1,2,figsize=(11.2,4),gridspec_kw={"width_ratios":[1.05,1.35]},layout="constrained")
    axes[0].scatter(gt-.07,gt-m,s=24,alpha=.72,color="#3F8797",label="MobileNetV3-FPN")
    axes[0].scatter(gt+.07,gt-r,s=23,alpha=.68,marker="D",color="#BF7954",label="ResNet50-FPN")
    axes[0].set(xlabel="Annotated pedestrians per original image",ylabel="Unmatched reference pedestrians",
                xlim=(.55,8.5),ylim=(-.14,2.5))
    axes[0].set_xticks(range(1,9));axes[0].set_yticks([0,1,2])
    axes[0].legend(fontsize=7.5,frameon=False,loc="upper left")
    y=np.arange(len(differences))
    misses_mobile=np.asarray([int(mobile[n]["gt"])-int(mobile[n]["matched"]) for n in differences])
    misses_resnet=np.asarray([int(resnet[n]["gt"])-int(resnet[n]["matched"]) for n in differences])
    axes[1].barh(y-.17,misses_mobile,height=.34,color="#3F8797",label="MobileNetV3-FPN")
    axes[1].barh(y+.17,misses_resnet,height=.34,color="#BF7954",label="ResNet50-FPN")
    axes[1].set(yticks=y,yticklabels=[Path(n).stem for n in differences],
                xlabel="Missed annotated pedestrians",xlim=(0,2.8),ylim=(-.6,len(differences)-.4))
    axes[1].set_xticks([0,1,2]);axes[1].invert_yaxis()
    axes[1].legend(fontsize=7.5,frameon=False,loc="lower right")
    for i,(u,v) in enumerate(zip(misses_mobile,misses_resnet)):
        axes[1].text(u+.05,i-.17,str(u),va="center",fontsize=8)
        axes[1].text(v+.05,i+.17,str(v),va="center",fontsize=8)
    for ax in axes:
        ax.grid(alpha=.14)
        ax.set_axisbelow(True)
        ax.spines[["top","right"]].set_visible(False)
    a.out.parent.mkdir(exist_ok=True,parents=True)
    for ext in ("pdf","png","svg"):
        fig.savefig(a.out.with_suffix("."+ext),dpi=210,bbox_inches="tight")
    print("Validated: 170 original paired photos, 166 ties and 4 differing photos",flush=True)

if __name__=="__main__":
    main()
