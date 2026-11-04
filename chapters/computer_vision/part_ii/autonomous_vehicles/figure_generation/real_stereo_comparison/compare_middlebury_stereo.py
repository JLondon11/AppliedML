"""Chapter 5 real Middlebury motorcycle stereo inference: OpenCV BM vs SGBM.

Run: python compare_middlebury_stereo.py
Inputs: scikit-image data.stereo_motorcycle(), including reference disparity.
Outputs: stereo_method_metrics.csv and Figures/real_stereo_comparison.{pdf,png,svg}.
This is an actual classical-stereo experiment, not RAFT, SAM, NeRF, or
camera–LiDAR fusion inference.
"""
from pathlib import Path
import csv
import time
import cv2
import numpy as np
from skimage import data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
(OUT/'Figures').mkdir(exist_ok=True)
left,right,gt=data.stereo_motorcycle()
gray_l=cv2.cvtColor(left,cv2.COLOR_RGB2GRAY)
gray_r=cv2.cvtColor(right,cv2.COLOR_RGB2GRAY)
reference=np.isfinite(gt)&(gt>0)&(gt<96)
assert np.sum(reference)>100000
bm=cv2.StereoBM_create(numDisparities=96,blockSize=15)
bm.setTextureThreshold(10)
bm.setUniquenessRatio(15)
sgbm=cv2.StereoSGBM_create(minDisparity=0,numDisparities=96,blockSize=7,
    P1=8*7**2,P2=32*7**2,disp12MaxDiff=1,uniquenessRatio=10,
    speckleWindowSize=80,speckleRange=2,preFilterCap=31,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
methods={'StereoBM':bm,'StereoSGBM':sgbm}
preds={};stats=[]
for name,method in methods.items():
    times=[]
    for i in range(4):
        t0=time.perf_counter()
        disp=method.compute(gray_l,gray_r).astype(np.float32)/16.
        duration_ms=(time.perf_counter()-t0)*1000
        if i:times.append(duration_ms)
    valid=reference & (disp>0)&(disp<96)
    errors=np.abs(disp[valid]-gt[valid])
    assert len(errors)>5000
    cover=float(valid.sum()/reference.sum())
    mae=float(errors.mean())
    rmse=float(np.sqrt(np.mean(errors*errors)))
    bad3_valid=float(np.mean(errors>3.))
    bad3_all=(np.sum(errors>3)+np.sum(reference&~valid))/np.sum(reference)
    stats.append((name,mae,rmse,cover,bad3_valid,float(bad3_all),
                  float(np.median(times))))
    preds[name]=(disp,valid)
with (OUT/'stereo_method_metrics.csv').open('w',newline='') as f:
    w=csv.writer(f)
    w.writerow(['method','MAE_px_predicted_valid','RMSE_px_predicted_valid',
       'coverage_of_reference_valid','bad3_rate_predicted_valid',
       'bad3_rate_reference_valid_including_missing',
       'median_runtime_ms_3_warm_runs'])
    w.writerows(stats)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
  'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axes=plt.subplots(2,3,figsize=(11,6.3),facecolor='white')
axes[0,0].imshow(left);axes[0,0].axis('off')
axes[0,1].imshow(np.ma.masked_where(~reference,gt),
                 cmap='viridis',vmin=0,vmax=85)
axes[0,1].axis('off')
axes[0,2].imshow(np.ma.masked_where(~preds['StereoBM'][1],
                                   preds['StereoBM'][0]),
                 cmap='viridis',vmin=0,vmax=85)
axes[0,2].axis('off')
axes[1,0].imshow(np.ma.masked_where(~preds['StereoSGBM'][1],
                                   preds['StereoSGBM'][0]),
                 cmap='viridis',vmin=0,vmax=85)
axes[1,0].axis('off')
abs_err=np.abs(preds['StereoSGBM'][0]-gt)
axes[1,1].imshow(np.ma.masked_where(~preds['StereoSGBM'][1],abs_err),
                 cmap='magma',vmin=0,vmax=12)
axes[1,1].axis('off')
x=np.arange(2)
axes[1,2].bar(x,[v[5]*100 for v in stats],
              color=['#b66a3f','#128b89'],width=.55)
axes[1,2].set_ylim(0,100)
axes[1,2].set_ylabel('Bad-3 including missing (%)')
axes[1,2].set_xticks(x,['StereoBM','StereoSGBM'],rotation=10)
axes[1,2].spines[['top','right']].set_visible(False)
for j,ax in enumerate(axes.flat):
    ax.text(.02,-.08,f'({"abcdef"[j]})',transform=ax.transAxes,
            color='black',fontsize=10)
fig.subplots_adjust(left=.02,right=.98,bottom=.09,top=.98,
                    wspace=.1,hspace=.22)
for ext in ('pdf','png','svg'):
    fig.savefig(OUT/'Figures'/f'real_stereo_comparison.{ext}',
                dpi=220,facecolor='white',bbox_inches='tight',
                pad_inches=.12)
plt.close(fig)
print('Reference-valid pixels:',int(reference.sum()))
for row in stats:print(row)
