"""Executed real Middlebury motorcycle stereo demonstration (not camera-LiDAR fusion).

Outputs actual StereoBM/SGBM predictions, reference-mask metrics, image-hash
provenance and a six-panel non-bar scientific plot. One original scene only.
"""
from pathlib import Path
import csv,hashlib,json,platform,time
import cv2,numpy as np,skimage
from skimage import data
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
left,right,gt=data.stereo_motorcycle()
assert left.shape==right.shape and gt.shape==left.shape[:2]
assert left.dtype==right.dtype==np.uint8
ref=np.isfinite(gt)&(gt>0)&(gt<96)
assert ref.sum()>100000
gl=cv2.cvtColor(left,cv2.COLOR_RGB2GRAY)
gr=cv2.cvtColor(right,cv2.COLOR_RGB2GRAY)
bm=cv2.StereoBM_create(numDisparities=96,blockSize=15)
bm.setTextureThreshold(10)
bm.setUniquenessRatio(15)
sgbm=cv2.StereoSGBM_create(minDisparity=0,numDisparities=96,
 blockSize=7,P1=8*7**2,P2=32*7**2,disp12MaxDiff=1,
 uniquenessRatio=10,speckleWindowSize=80,speckleRange=2,preFilterCap=31,
 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
rows=[];preds={};error_vectors={}
for name,method in (("StereoBM",bm),("StereoSGBM",sgbm)):
    times=[]
    for j in range(4):
        start=time.perf_counter()
        disp=method.compute(gl,gr).astype(np.float32)/16
        elapsed=(time.perf_counter()-start)*1000
        if j:times.append(elapsed)
    valid=ref&(disp>0)&(disp<96)
    err=np.abs(disp[valid]-gt[valid])
    missing=int((ref&~valid).sum())
    rows.append({
      "method":name,
      "reference_valid_pixels":int(ref.sum()),
      "evaluated_predicted_valid_pixels":int(valid.sum()),
      "missing_reference_valid_pixels":missing,
      "coverage":float(valid.sum()/ref.sum()),
      "mae_valid_px":float(err.mean()),
      "rmse_valid_px":float(np.sqrt(np.mean(err**2))),
      "bad3_predicted_valid":float(np.mean(err>3)),
      "bad3_all_reference_including_missing":float(((err>3).sum()+missing)/ref.sum()),
      "median_warm_runtime_ms":float(np.median(times))})
    preds[name]=(disp,valid)
    error_vectors[name]=err
with (OUT/"original_measurements.csv").open("w",newline="") as handle:
    writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows)
provenance={"dataset":"scikit-image data.stereo_motorcycle original Middlebury stereo motorcycle photo pair and reference disparity",
  "skimage_version":skimage.__version__,"opencv_version":cv2.__version__,
  "numpy_version":np.__version__,"python_version":platform.python_version(),
  "source_pixel_hashes_sha256":{name:hashlib.sha256(value.tobytes()).hexdigest()
      for name,value in (("left",left),("right",right),("reference",gt))},
  "reference_valid_pixels":int(ref.sum()),
  "scope":"Single authentic stereo image pair, no broader benchmark or camera-LiDAR fusion",
  "metrics":rows}
(OUT/"evidence_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n",encoding="utf-8")
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,
    "pdf.fonttype":42,"svg.fonttype":"none"})
navy="#21364e";teal="#087f80";copper="#bc6a4a"
fig=plt.figure(figsize=(11.8,7.9),facecolor="white")
gs=fig.add_gridspec(2,3,left=.045,right=.99,top=.89,bottom=.14,wspace=.16,hspace=.26)
ax=[fig.add_subplot(gs[i,j]) for i in range(2) for j in range(3)]
ax[0].imshow(left)
ax[0].set_title("A  Original left camera photograph",loc="left",weight="bold",color=navy)
ax[1].imshow(np.ma.masked_where(~ref,gt),cmap="viridis",vmin=0,vmax=90)
ax[1].set_title("B  Measured reference disparity",loc="left",weight="bold",color=navy)
for name,idx in (("StereoBM",2),("StereoSGBM",3)):
    disp,valid=preds[name]
    ax[idx].imshow(np.ma.masked_where(~valid,disp),cmap="viridis",vmin=0,vmax=90)
    ax[idx].set_title(("C  " if idx==2 else "D  ")+name+" prediction",
       loc="left",weight="bold",color=navy)
disparity,valid=preds["StereoSGBM"]
ax[4].imshow(np.ma.masked_where(~(valid&ref),np.abs(disparity-gt)),cmap="magma",vmin=0,vmax=12)
ax[4].set_title("E  SGBM absolute disparity error",loc="left",weight="bold",color=navy)
for a in ax[:5]:a.set_axis_off()
x=np.linspace(0,20,300)
for row,color in zip(rows,[copper,teal]):
    errors=error_vectors[row["method"]]
    ecdf=np.searchsorted(np.sort(errors),x,side="right")/len(errors)
    ax[5].plot(x,ecdf,label=f"{row['method']}  (MAE {row['mae_valid_px']:.2f} px)",
      color=color,lw=2)
ax[5].set(xlim=(0,20),ylim=(0,1),xlabel="Absolute disparity error (pixels)",
  ylabel="Fraction of valid predictions ≤ error")
ax[5].grid(color="#e2e8ed")
ax[5].set_title("F  Measured error distributions",loc="left",weight="bold",color=navy)
ax[5].legend(frameon=False,fontsize=8,loc="lower right")
fig.suptitle("Real Middlebury stereo scene · measured disparity estimation",
 x=.045,y=.97,ha="left",fontsize=16,color=navy,weight="bold")
fig.text(.045,.915,f"Original {left.shape[1]} × {left.shape[0]} photographs · {ref.sum():,} valid reference pixels · StereoBM vs StereoSGBM",
 color="#4a6172",fontsize=9.7)
fig.text(.045,.091,f"StereoBM: coverage {rows[0]['coverage']*100:.1f}% · all-ref bad-3 {rows[0]['bad3_all_reference_including_missing']*100:.1f}% · measured warm runtime {rows[0]['median_warm_runtime_ms']:.1f} ms",fontsize=9,color=navy)
fig.text(.045,.069,f"StereoSGBM: coverage {rows[1]['coverage']*100:.1f}% · all-ref bad-3 {rows[1]['bad3_all_reference_including_missing']*100:.1f}% · measured warm runtime {rows[1]['median_warm_runtime_ms']:.1f} ms",fontsize=9,color=navy)
fig.text(.045,.046,"Disparity images: shared 0–90-pixel viridis scale · error panel: 0–12-pixel magma scale.",fontsize=8.7,color="#566776")
fig.text(.045,.025,"One authentic scene; error ECDF excludes missing estimates (counted as failures in all-reference bad-3).",fontsize=8.5,color="#566776")
for ext in ("pdf","png","svg"):
    fig.savefig(OUT/f"real_middlebury_stereo_comparison.{ext}",facecolor="white",dpi=225)
plt.close(fig)
print(json.dumps(rows,indent=2))
