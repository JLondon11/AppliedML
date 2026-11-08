"""Figure 5.48: genuine 9-iteration OpenCV GrabCut and distinct isolated foreground."""
from pathlib import Path
import csv
import numpy as np
import cv2
from skimage import data
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"Figures";OUT.mkdir(exist_ok=True)
img=data.astronaut();h,w=img.shape[:2];rect=(112,15,285,470)
seg=np.zeros((h,w),np.uint8);bg=np.zeros((1,65),np.float64);fg=np.zeros((1,65),np.float64)
cv2.grabCut(cv2.cvtColor(img,cv2.COLOR_RGB2BGR),seg,rect,bg,fg,9,cv2.GC_INIT_WITH_RECT)
mask=np.isin(seg,[cv2.GC_FGD,cv2.GC_PR_FGD]).astype(np.uint8)
assert int(mask.sum())==62948
cutout=np.full_like(img,225)
cutout[mask.astype(bool)]=img[mask.astype(bool)]
fig,axs=plt.subplots(1,3,figsize=(11,3.9),facecolor="white")
axs[0].imshow(img);axs[0].add_patch(Rectangle((rect[0],rect[1]),rect[2],rect[3],ec="#da721e",fc="none",lw=2))
axs[1].imshow(mask,cmap="gray",vmin=0,vmax=1)
axs[2].imshow(cutout);axs[2].contour(mask,levels=[.5],colors=["#0b8479"],linewidths=1)
for i,ax in enumerate(axs):
 ax.axis("off");ax.text(.5,-.065,f"({'abc'[i]})",transform=ax.transAxes,ha="center",fontsize=10)
fig.subplots_adjust(left=.02,right=.98,bottom=.14,top=.98,wspace=.025)
for ext in ("png","pdf","svg"):
 fig.savefig(OUT/f"real_image_grabcut_baseline.{ext}",dpi=220,bbox_inches="tight",pad_inches=.08,facecolor="white")
plt.close(fig)
with (OUT/"grabcut_figure_548_metrics.csv").open("w",newline="") as f:
 writer=csv.writer(f);writer.writerow(["metric","value"]);writer.writerow(["foreground_pixels",int(mask.sum())]);writer.writerow(["mask_fraction",float(mask.mean())])
