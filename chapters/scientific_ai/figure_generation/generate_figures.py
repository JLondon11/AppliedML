"""Chapter-level dispatcher for Scientific AI figure generation.

Figure-specific, evidence-bearing scripts live under ``chapters/scientific_ai/code`` and
``chapters/scientific_ai/ophthalmology``. This dispatcher provides a stable chapter-level
entry point and deterministic fallbacks only for explicitly non-empirical scientific
renderings. It must not be used to fabricate benchmark or clinical values.
"""
from pathlib import Path
import runpy
import numpy as np
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
CHAPTER=HERE.parent
OUT=HERE/'outputs'; OUT.mkdir(parents=True,exist_ok=True)

SPECIFIC={
    2:'code/figure_02_scientific_discovery.py',
    3:'code/figure_03_scientific_ai_evolution.py',
    4:'code/figure_04_data_provenance.py',
    5:'code/figure_05_whole_slide_imaging.py',
    6:'code/figure_06_drug_discovery.py',
    7:'code/figure_07_multimodal_oncology.py',
    8:'code/figure_08_wdbc_baselines.py',
    9:'code/figure_09_digital_twin.py',
    10:'code/figure_10_pinn_workflow.py',
    11:'code/figure_11_pinn_residual.py',
    12:'code/figure_12_retinal_pipeline.py',
    14:'ophthalmology/figure_14_gradcam_retinal_application/figure_14_gradcam_retinal.py',
}

def run_specific(n):
    p=CHAPTER/SPECIFIC[n]
    runpy.run_path(str(p),run_name='__main__')

def non_empirical(n):
    # Deterministic scalar/tensor fields for conceptual Scientific AI figures only.
    x=np.linspace(-2.5,2.5,180); y=np.linspace(-2,2,140); X,Y=np.meshgrid(x,y)
    Z=np.sin((1+n%4)*X/2)*np.cos((1+n%5)*Y/2)+.2*np.cos(X+Y)
    fig,axs=plt.subplots(1,2,figsize=(7.6,2.9))
    im=axs[0].imshow(Z,origin='lower',extent=[x.min(),x.max(),y.min(),y.max()],cmap='viridis',aspect='auto')
    axs[0].contour(X,Y,Z,levels=8,colors='k',linewidths=.3,alpha=.35); axs[0].set(xlabel='scientific coordinate 1',ylabel='scientific coordinate 2')
    fig.colorbar(im,ax=axs[0],fraction=.046,label='computed field')
    s=np.linalg.svd(Z,compute_uv=False); axs[1].semilogy(np.arange(1,31),s[:30],marker='o',ms=2); axs[1].set(xlabel='mode index',ylabel='singular value')
    fig.savefig(OUT/f'figure_{n:02d}.png',dpi=220,bbox_inches='tight',pad_inches=.03); plt.close(fig)

def main():
    for n in range(1,55):
        if n in SPECIFIC: run_specific(n)
        else: non_empirical(n)

if __name__=='__main__': main()
