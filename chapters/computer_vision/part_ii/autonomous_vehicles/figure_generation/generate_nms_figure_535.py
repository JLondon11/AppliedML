"""Regenerate Figure 5.35: measured output of class-aware nonmaximum suppression."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BOXES=np.array([[.55,.60,2.55,2.55],[.82,.83,2.77,2.72],
                [3.25,.65,4.85,2.35],[.94,.95,2.60,2.65]])
SCORES=np.array([.94,.81,.87,.72])
CLASSES=np.array([0,0,0,1])
NAMES="ABCD"

def iou(a,b):
    lo=np.maximum(a[:2],b[:2])
    hi=np.minimum(a[2:],b[2:])
    intersection=np.prod(np.maximum(0,hi-lo))
    return float(intersection/(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection))

def class_aware_nms(boxes,scores,classes,threshold=.5):
    keep=[]
    for cls in np.unique(classes):
        order=np.flatnonzero(classes==cls)
        order=order[np.argsort(-scores[order],kind="stable")]
        while len(order):
            chosen=int(order[0]);keep.append(chosen)
            order=np.array([j for j in order[1:] if iou(boxes[chosen],boxes[j])<=threshold],dtype=int)
    return sorted(keep,key=lambda j:-scores[j])

keep=class_aware_nms(BOXES,SCORES,CLASSES)
assert keep==[0,2,3]
assert iou(BOXES[0],BOXES[1])>.5
assert iou(BOXES[0],BOXES[3])>.5

colors=["#16718a","#d65c45","#3a8765","#9064b3"]
fig,axes=plt.subplots(1,3,figsize=(12.3,3.9),gridspec_kw={"width_ratios":[1.12,.90,1.12]},layout="constrained")

def draw_boxes(ax,indices,title):
    positions=[(.45,2.31),(2.13,2.89),(3.28,2.23),(.90,3.11)]
    for j in indices:
        x,y,x2,y2=BOXES[j]
        ax.add_patch(Rectangle((x,y),x2-x,y2-y,fc="none",ec=colors[j],
                               lw=2.5,ls="--" if j==1 else "-"))
        tx,ty=positions[j]
        ax.text(tx,ty,f"{NAMES[j]}: {SCORES[j]:.2f}",color=colors[j],weight="bold",
                fontsize=9,bbox={"facecolor":"white","alpha":.9,"edgecolor":"none","pad":.7})
    ax.set(xlim=(.18,5.13),ylim=(.17,3.40),xlabel="Image x",ylabel="Image y")
    ax.set_aspect("equal")
    ax.grid(alpha=.12)
    ax.set_title(title,fontsize=11)

draw_boxes(axes[0],[0,1,2,3],"(a) Candidate boxes")
axes[0].text(.23,.32,"A, B, C: class 0; D: class 1",fontsize=8.4,
             bbox={"facecolor":"white","edgecolor":"none","alpha":.9})
overlaps=[iou(BOXES[0],BOXES[1]),iou(BOXES[0],BOXES[3])]
axes[1].barh([1,0],overlaps,color=[colors[1],colors[3]],height=.38)
axes[1].axvline(.5,color="#444444",ls="--",lw=1.5)
for y,v in zip([1,0],overlaps):
    axes[1].text(v+.025,y,f"{v:.2f}",va="center",fontsize=9)
axes[1].set(xlim=(0,1),ylim=(-.7,1.8),yticks=[0,1],
            yticklabels=["A–D (different class)","A–B (same class)"],
            xlabel="Intersection over union")
axes[1].set_title("(b) Pairwise overlap",fontsize=11)
axes[1].grid(axis="x",alpha=.15)
draw_boxes(axes[2],keep,"(c) Retained detections")
axes[2].text(.23,.32,"B suppressed; A, C, D retained",fontsize=8.4,
             bbox={"facecolor":"white","edgecolor":"none","alpha":.9})

output=Path(__file__).resolve().parents[1]/"Figures"
output.mkdir(exist_ok=True)
for extension in ("png","pdf","svg"):
    fig.savefig(output/f"nms_verified_output.{extension}",dpi=220,
                facecolor="white",bbox_inches="tight",pad_inches=.06)
plt.close(fig)
with (output/"nms_figure_535_metrics.csv").open("w",newline="") as handle:
    writer=csv.writer(handle)
    writer.writerow(["candidate","class","score","retained","iou_with_A"])
    for j in range(4):
        writer.writerow([NAMES[j],int(CLASSES[j]),float(SCORES[j]),
                         int(j in keep),round(iou(BOXES[0],BOXES[j]),6)])
print("retained",keep,"IoU A-B",overlaps[0],"IoU A-D",overlaps[1])
