"""Chapter 5 real-photograph computational baselines: GrabCut and StereoSGBM.
Neither figure is SAM, NeRF, or 3D Gaussian Splatting inference.
Run python generate_real_photo_baselines.py (numpy/opencv-python/scikit-image/matplotlib).
"""
from pathlib import Path
import csv
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import data
from matplotlib.patches import Rectangle
ROOT=Path(__file__).resolve().parent
FIG=ROOT/"Figures";FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,
"axes.labelsize":9,"pdf.fonttype":42,"svg.fonttype":"none"})
# Real scikit-image astronaut photograph (not an annotation benchmark).
img=data.astronaut()
H,W=img.shape[:2]
rect=(112,15,285,470)
bgr=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
seg=np.zeros((H,W),np.uint8)
bg=np.zeros((1,65),np.float64);fg=np.zeros((1,65),np.float64)
cv2.grabCut(bgr,seg,rect,bg,fg,9,cv2.GC_INIT_WITH_RECT)
mask=np.where((seg==2)|(seg==0),0,1).astype(np.uint8)
assert mask.shape==img.shape[:2] and 10000<mask.sum()<200000
fig,axes=plt.subplots(1,3,figsize=(11,3.9),facecolor="white")
axes[0].imshow(img)
axes[0].add_patch(Rectangle((rect[0],rect[1]),rect[2],rect[3],
                        fill=False,edgecolor="#da721e",lw=2))
axes[1].imshow(mask,cmap="gray",vmin=0,vmax=1)
axes[2].imshow(img)
axes[2].imshow(np.ma.masked_where(mask==0,mask),
   cmap=matplotlib.colors.ListedColormap(["#0a9e8d"]),alpha=.32,vmin=0,vmax=1)
axes[2].contour(mask,levels=[.5],colors=["#0b8479"],linewidths=.9)
for i,ax in enumerate(axes):
    ax.set_axis_off()
    ax.text(.5,-.065,"("+"abc"[i]+")",transform=ax.transAxes,
            ha="center",color="black",fontsize=10)
fig.subplots_adjust(left=.02,right=.98,bottom=.14,top=.98,wspace=.025)
for ext in ("png","pdf","svg"):
    fig.savefig(FIG/f"real_image_grabcut_baseline.{ext}",
                dpi=220,bbox_inches="tight",pad_inches=.08,facecolor="white")
plt.close(fig)

# Real Middlebury rectified motorcycle stereo pair + supplied disparity ground truth.
left,right,reference=data.stereo_motorcycle()
gl=cv2.cvtColor(left,cv2.COLOR_RGB2GRAY)
gr=cv2.cvtColor(right,cv2.COLOR_RGB2GRAY)
sgbm=cv2.StereoSGBM_create(minDisparity=0,numDisparities=96,blockSize=7,
 P1=8*7**2,P2=32*7**2,disp12MaxDiff=1,uniquenessRatio=10,
 speckleWindowSize=80,speckleRange=2,preFilterCap=31,
 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
disp=sgbm.compute(gl,gr).astype(np.float32)/16
valid=np.isfinite(reference)&(reference>0)&(disp>0)&(disp<96)
err=np.abs(disp-reference)
assert valid.sum()>50000
mae=float(np.mean(err[valid]))
rmse=float(np.sqrt(np.mean(err[valid]**2)))
coverage=float(valid.mean())
fig,axes=plt.subplots(1,3,figsize=(11.6,3.35),facecolor="white")
axes[0].imshow(left)
axes[1].imshow(np.ma.masked_where(disp<=0,disp),cmap="viridis",vmin=0,vmax=70)
axes[2].imshow(np.ma.masked_where(~valid,err),cmap="magma",vmin=0,vmax=15)
for i,ax in enumerate(axes):
    ax.axis("off")
    ax.text(.5,-.063,"("+"abc"[i]+")",transform=ax.transAxes,
            ha="center",color="black",fontsize=10)
fig.subplots_adjust(left=.01,right=.99,bottom=.13,top=.98,wspace=.02)
for ext in ("png","pdf","svg"):
    fig.savefig(FIG/f"real_stereo_sgbm_baseline.{ext}",
                dpi=220,bbox_inches="tight",pad_inches=.07,facecolor="white")
plt.close(fig)
with (ROOT/"real_photo_baseline_metrics.csv").open("w",newline="") as f:
    writer=csv.writer(f);writer.writerow(["metric","value","basis"])
    writer.writerow(["grabcut_mask_foreground_pixels",int(mask.sum()),
        "real scikit-image astronaut photograph; fixed rectangle prompt"])
    writer.writerow(["stereo_sgbm_mae_pixels",mae,
        "Middlebury stereo motorcycle: valid reference and prediction"])
    writer.writerow(["stereo_sgbm_rmse_pixels",rmse,
        "Middlebury stereo motorcycle: valid reference and prediction"])
    writer.writerow(["stereo_evaluation_valid_fraction",coverage,
        "fraction of reference and prediction valid pixels"])
print("GrabCut mask",int(mask.sum()),"SGBM MAE",mae,
      "SGBM RMSE",rmse,"coverage",coverage)
