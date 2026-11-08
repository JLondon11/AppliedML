"""Generate real TuSimple annotation-geometry figure (not trained-model predictions).

Official publicly documented annotation sample:
https://github.com/TuSimple/tusimple-benchmark/blob/master/doc/lane_detection/readme.md
"""
import csv
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Exactly the four documented example lane traces, with -2 denoting missing points.
lanes = [
[-2,-2,-2,-2,632,625,617,609,601,594,586,578,570,563,555,547,539,532,524,516,508,501,493,485,477,469,462,454,446,438,431,423,415,407,400,392,384,376,369,361,353,345,338,330,322,314,307,299],
[-2,-2,-2,-2,719,734,748,762,777,791,805,820,834,848,863,877,891,906,920,934,949,963,978,992,1006,1021,1035,1049,1064,1078,1092,1107,1121,1135,1150,1164,1178,1193,1207,1221,1236,1250,1265,-2,-2,-2,-2,-2],
[-2,-2,-2,-2,-2,532,503,474,445,416,387,358,329,300,271,241,212,183,154,125,96,67,38,9,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2],
[-2,-2,-2,781,822,862,903,944,984,1025,1066,1107,1147,1188,1229,1269,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2]
]
y = np.arange(240,711,10)
assert all(len(row)==len(y) for row in lanes)
h,w = 720,1280
mask = np.zeros((h,w),dtype=np.uint8)
fig,axes = plt.subplots(1,3,figsize=(12.8,3.9),layout="constrained")
colors = ["#227f9c","#e58935","#8d5da5","#39815f"]
results = []
for n,values in enumerate(lanes,1):
    x = np.asarray(values,dtype=int)
    visible = (x>=0)&(x<w)
    xv,yv = x[visible],y[visible]
    assert xv.size>=2
    coords = np.stack([xv,yv],axis=1).astype(np.int32)
    cv2.polylines(mask,[coords],False,255,thickness=4)
    axes[0].plot(xv,yv,marker=".",lw=1.7,ms=3,color=colors[n-1],label=f"Lane {n}")
    dxdy = np.diff(xv)/np.diff(yv)
    axes[2].plot((yv[:-1]+yv[1:])/2,dxdy,marker=".",lw=1.4,ms=2,color=colors[n-1])
    results.append((n,len(xv),int(yv.min()),int(yv.max()),int(xv.min()),int(xv.max()),float(np.mean(abs(dxdy)))))
axes[0].set(xlim=(0,w),ylim=(h,220),xlabel="Horizontal position x (pixels)",ylabel="Image row y (pixels)",title="(a) Annotated lane coordinates")
axes[0].legend(fontsize=7)
axes[0].grid(alpha=.15)
axes[1].imshow(mask,cmap="gray",vmin=0,vmax=255)
axes[1].set(xlim=(0,w),ylim=(h,220),xlabel="Image x (pixels)",ylabel="Image y (pixels)",title="(b) Rasterized annotation (4-pixel lines)")
axes[2].set(xlim=(235,715),xlabel="Image row y (pixels)",ylabel="Local horizontal displacement dx/dy",title="(c) Annotated lane slope")
axes[2].axhline(0,color="gray",lw=.7)
axes[2].grid(alpha=.15)
out = Path(__file__).resolve().parent
for ext in ("png","pdf","svg"):
    fig.savefig(out/f"tusimple_annotation_geometry.{ext}",dpi=220,bbox_inches="tight",facecolor="white")
plt.close(fig)
with (out/"tusimple_annotation_geometry_metrics.csv").open("w",newline="") as f:
    writer=csv.writer(f)
    writer.writerow(("lane","valid_points","first_row","last_row","min_x","max_x","mean_abs_dxdy"))
    writer.writerows(results)
print("Counts:",[r[1] for r in results],"Mask foreground pixels:",int((mask>0).sum()))
