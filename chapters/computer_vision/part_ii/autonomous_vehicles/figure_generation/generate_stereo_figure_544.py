"""Figure 5.44: actual distinct Middlebury stereo motorcycle photographs and CPU StereoSGBM."""
from pathlib import Path
import csv
import cv2
import numpy as np
from skimage import data
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"Figures"
OUT.mkdir(parents=True,exist_ok=True)
left,right,reference=data.stereo_motorcycle()
assert left.shape==right.shape and not np.array_equal(left,right)
gl=cv2.cvtColor(left,cv2.COLOR_RGB2GRAY)
gr=cv2.cvtColor(right,cv2.COLOR_RGB2GRAY)
block=7
model=cv2.StereoSGBM_create(minDisparity=0,numDisparities=96,blockSize=block,
 P1=8*block**2,P2=32*block**2,disp12MaxDiff=1,uniquenessRatio=10,
 speckleWindowSize=80,speckleRange=2,preFilterCap=31,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
disparity=model.compute(gl,gr).astype(np.float32)/16.0
valid=np.isfinite(disparity)&(disparity>0)
fig,axes=plt.subplots(1,3,figsize=(12.5,3.75),gridspec_kw={"width_ratios":[1,1,1.07]})
for ax,img,title in zip(axes[:2],[left,right],["(a) Left camera","(b) Right camera"]):
 ax.imshow(img);ax.set_axis_off();ax.set_title(title,fontsize=11,pad=4)
 ax.add_patch(Rectangle((330,120),165,145,edgecolor="#ffcf57",facecolor="none",lw=1.8))
im=axes[2].imshow(np.ma.masked_where(~valid,disparity),cmap="viridis",vmin=0,vmax=96)
axes[2].set_axis_off();axes[2].set_title("(c) StereoSGBM disparity",fontsize=11,pad=4)
bar=fig.colorbar(im,ax=axes[2],fraction=.043,pad=.02)
bar.set_label("Disparity (pixels)",fontsize=10)
fig.subplots_adjust(left=.01,right=.99,top=.90,bottom=.02,wspace=.06)
for ext in ("pdf","png","svg"):
 fig.savefig(OUT/f"stereo_sgbm_executed.{ext}",dpi=220,facecolor="white",bbox_inches="tight",pad_inches=.07)
plt.close(fig)
with (OUT/"stereo_figure_544_metrics.csv").open("w",newline="") as fh:
 w=csv.writer(fh);w.writerow(["metric","value","definition"])
 w.writerow(["positive_disparity_fraction",f"{valid.mean():.8f}","positive StereoSGBM pixels / all image pixels"])
 w.writerow(["left_right_rgb_mae",f"{np.mean(np.abs(left.astype(float)-right.astype(float))):.8f}","mean absolute RGB difference of separate original images"])
print("Positive disparity pixel fraction:",valid.mean())
