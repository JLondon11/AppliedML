"""Figure 5.43: analytic phantom Fourier undersampling, no patient data or trained model."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from skimage.metrics import structural_similarity
out=Path(__file__).resolve().parent
phantom=resize(shepp_logan_phantom(),(256,256),order=1,mode="reflect",anti_aliasing=True)
mask=np.zeros((256,256),dtype=bool);mask[::4,:]=True;mask[120:136,:]=True
k=np.fft.fftshift(np.fft.fft2(phantom))
recon=np.abs(np.fft.ifft2(np.fft.ifftshift(k*mask)))
err=np.abs(recon-phantom)
rmse=float(np.sqrt(np.mean((recon-phantom)**2))/np.ptp(phantom))
ssim=float(structural_similarity(phantom,recon,data_range=float(np.ptp(phantom))))
assert mask.mean()==.296875 and abs(rmse-.104)<.003 and abs(ssim-.535)<.005
fig,axs=plt.subplots(1,4,figsize=(12.8,3.5),gridspec_kw={"width_ratios":[1,1,1,1.15]},layout="constrained")
for a,img,title in zip(axs[:3],[phantom,mask,recon],["(a) Analytical phantom","(b) Fourier sampling mask","(c) Zero-filled magnitude"]):
 a.imshow(img,cmap="gray",vmin=0,vmax=1);a.axis("off");a.set_title(title,fontsize=10)
im=axs[3].imshow(err,cmap="magma",vmin=0);axs[3].axis("off");axs[3].set_title("(d) Absolute error",fontsize=10)
fig.colorbar(im,ax=axs[3],fraction=.044,pad=.02,label="Absolute intensity error")
for ext in ("png","pdf","svg"):
 fig.savefig(out/f"fourier_phantom_executed.{ext}",dpi=220,bbox_inches="tight",pad_inches=.04)
plt.close(fig)
with (out/"fourier_phantom_543_metrics.csv").open("w",newline="") as f:
 w=csv.writer(f);w.writerow(["sampling_fraction","range_normalized_rmse","ssim"])
 w.writerow([mask.mean(),rmse,ssim])
print("Fraction",mask.mean(),"range normalized RMSE",rmse,"SSIM",ssim)
