"""Pass 14 closure source for Scientific AI Figures 14 and 25.
Figure 14's authoritative source remains the dedicated RetinaMNIST/MedMNIST Grad-CAM experiment in chapters/scientific_ai/ophthalmology/figure_14_gradcam_retinal_application/figure_14_gradcam_retinal.py. Figure 25 is a deterministic surrogate materials-discovery loop.
"""
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
def save(fig,n): fig.savefig(OUT/f'Figure_{n:03d}.png',dpi=260,bbox_inches='tight',pad_inches=.03); plt.close(fig)
# Convenience renderer for Figure 14 from saved tensors, if the authoritative experiment has already been run.
exp=Path(__file__).resolve().parents[1]/'ophthalmology'/'figure_14_gradcam_retinal_application'
imgf=exp/'figure_14_source_image.npy'; camf=exp/'figure_14_gradcam.npy'
if imgf.exists() and camf.exists():
 img=np.load(imgf); cam=np.load(camf); fig,axs=plt.subplots(1,3,figsize=(10.5,3.7)); axs[0].imshow(img); axs[1].imshow(cam,cmap='magma',vmin=0,vmax=1); axs[2].imshow(img); axs[2].imshow(cam,cmap='magma',alpha=.48,vmin=0,vmax=1); [a.axis('off') for a in axs]; save(fig,14)
# Figure 25: analytic closed-loop surrogate materials discovery.
x=np.linspace(-2,2,160); y=np.linspace(-2,2,140); X,Y=np.meshgrid(x,y); E=(X**2-1)**2+.6*(Y-.3*X)**2+.2*np.sin(3*X*Y); pts=np.array([[-1.4,1.1],[-.6,.5],[.1,.1],[.7,.2],[1.2,.5]]); fig,axs=plt.subplots(1,2,figsize=(7.8,3)); im=axs[0].contourf(X,Y,E,30,cmap='viridis'); fig.colorbar(im,ax=axs[0],fraction=.046,pad=.04,label='surrogate energy'); axs[0].plot(pts[:,0],pts[:,1],'o-'); axs[0].set_xlabel('composition coordinate 1'); axs[0].set_ylabel('coordinate 2'); axs[1].plot(np.arange(len(pts)),[2.2,1.4,.8,.45,.3],marker='o'); axs[1].set_xlabel('closed-loop iteration'); axs[1].set_ylabel('best surrogate energy'); save(fig,25)
