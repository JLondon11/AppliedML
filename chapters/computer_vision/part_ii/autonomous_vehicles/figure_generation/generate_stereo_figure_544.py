"""Figure 5.44: source Middlebury left/right images and actual StereoSGBM disparity."""
from pathlib import Path
import csv
import numpy as np
import cv2
from skimage import data
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"Figures";OUT.mkdir(exist_ok=True)
left,right,reference=data.stereo_motorcycle()
assert left.shape==right.shape and not np.array_equal(left,right)
bl=cv2.cvtColor(left,cv2.COLOR_RGB2GRAY);br=cv2.cvtColor(right,cv2.COLOR_RGB2GRAY)
b=7
sgbm=cv2.StereoSGBM_create(minDisparity=0,numDisparities=96,blockSize=b,P1=8*b*b,P2=32*b*b,disp12MaxDiff=1,uniquenessRatio=10,speckleWindowSize=80,speckleRange=2,preFilterCap=31,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
disparity=sgbm.compute(bl,br).astype(np.float32)/16
valid=(disparity>0)&np.isfinite(disparity)
x,y,w,h=330,120,165,145
l_crop=left[y:y+h,x:x+w];r_crop=right[y:y+h,x:x+w]
difference=np.mean(abs(l_crop.astype(float)-r_crop.astype(float)),axis=2)
fig,axs=plt.subplots(2,3,figsize=(12.0,6.0),gridspec_kw={"height_ratios":[1.65,1]},layout="constrained")
for j,(im,label) in enumerate([(left,"(a) Left camera"),(right,"(b) Right camera")]):
 ax=axs[0,j];ax.imshow(im);ax.add_patch(Rectangle((x,y),w,h,ec="#f6ca48",fc="none",lw=2));ax.set_title(label,fontsize=11);ax.axis("off")
ax=axs[0,2];m=ax.imshow(np.ma.masked_where(~valid,disparity),cmap="viridis",vmin=0,vmax=96);ax.set_title("(c) Measured disparity",fontsize=11);ax.axis("off");fig.colorbar(m,ax=ax,shrink=.67,label="Disparity (pixels)")
for j,(im,label) in enumerate([(l_crop,"(d) Enlarged left crop"),(r_crop,"(e) Enlarged right crop")]):
 ax=axs[1,j];ax.imshow(im,interpolation="nearest");ax.set_title(label,fontsize=10);ax.axis("off")
ax=axs[1,2];m=ax.imshow(difference,cmap="magma",vmin=0,vmax=100);ax.set_title("(f) Same-coordinate crop RGB difference",fontsize=10);ax.axis("off");fig.colorbar(m,ax=ax,shrink=.78,label="Mean absolute channel difference")
for ext in ("png","pdf","svg"):fig.savefig(OUT/f"stereo_sgbm_executed.{ext}",dpi=220,bbox_inches="tight",pad_inches=.05,facecolor="white")
plt.close(fig)
metrics={"positive_disparity_fraction":float(valid.mean()),"left_right_rgb_mae":float(np.mean(abs(left.astype(float)-right.astype(float)))),"crop_rgb_mae":float(difference.mean()),"image_width":left.shape[1],"image_height":left.shape[0]}
with (ROOT/"code"/"stereo_figure_544_metrics.csv").open("w",newline="") as fh:
 writer=csv.writer(fh);writer.writerow(["metric","value"]);writer.writerows(metrics.items())
print(metrics)
