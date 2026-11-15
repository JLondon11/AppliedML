"""Make scientific spatial-error and error-distribution figures from REAL original test patches.

Requires real heldout_metrics.csv from this chapter's executed original
scikit-image immunohistochemistry CVAE. No artificial observations.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot(folder,out):
    folder=Path(folder);out=Path(out)
    with (folder/'heldout_metrics.csv').open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    m=json.loads((folder/'metrics.json').read_text())
    if len(rows)!=217 or m['n_test_patches']!=217 or m['source_images']!=1:
        raise ValueError('Original one-tissue study source size mismatch')
    x=np.array([int(r['col']) for r in rows])
    y=np.array([int(r['row']) for r in rows])
    psnr=np.array([float(r['recon_PSNR_dB']) for r in rows])
    ssim=np.array([float(r['recon_SSIM']) for r in rows])
    mse=np.array([float(r['recon_MSE']) for r in rows])
    base=np.array([float(r['nearest_training_patch_MSE']) for r in rows])
    if len(set(zip(x,y)))!=217 or np.any(y<384) or np.any(y+32>512):
        raise ValueError('Original test must use lower nonoverlapping tissue-source region')
    if not np.all(np.isfinite(np.r_[psnr,ssim,mse,base])):
        raise ValueError('Unexpected nonfinite original measured errors')
    for vals,key in ((psnr,'test_reconstruction_PSNR_dB'),(ssim,'test_reconstruction_SSIM'),
                     (mse,'test_reconstruction_MSE'),(base,'test_nearest_training_patch_MSE')):
        if abs(vals.mean()-m[key])>1e-7:raise ValueError('Original aggregated source metric mismatch '+key)
    out.mkdir(parents=True,exist_ok=True)
    fig,axs=plt.subplots(1,2,figsize=(11,5.25),layout='constrained')
    palette='#173e59'
    scatter=axs[0].scatter(x+16,y+16,c=psnr,s=85,marker='s',
                         cmap='viridis',vmin=np.percentile(psnr,5),vmax=np.percentile(psnr,95),
                         edgecolors='none')
    axs[0].set(xlim=(0,512),ylim=(520,380),xlabel='Position in original source photograph: column (pixels)',
        ylabel='Source-photo row (pixels)',title='A  Spatial variation in measured held-out PSNR')
    fig.colorbar(scatter,ax=axs[0],label='Original patch reconstruction PSNR (dB)',shrink=.77)
    order=np.sort(mse)
    baseline=np.sort(base)
    for values,color,name in ((order,'#187e83','VAE reconstructed held-out tissue'),
                              (baseline,'#ba6c4d','Nearest original training patch')):
        axs[1].step(values,np.arange(1,len(values)+1)/len(values),where='post',
                    color=color,lw=2,label=name)
    axs[1].set(xlabel='Per-patch mean squared pixel error',ylabel='Fraction of 217 test patches',
               title='B  Reconstruction-error distribution',ylim=(0,1.01))
    axs[1].legend(loc='lower right',frameon=False,fontsize=8)
    for ax in axs:
        ax.grid(color='#e5eaed',alpha=.8)
        ax.spines[['right','top']].set_visible(False)
    fig.suptitle('Original immunohistochemistry tissue · CVAE held-out reconstruction',
                 color=palette,fontweight='bold',fontsize=14)
    fig.text(.025,.01,'One genuine parent photograph; 217 spatially distinct but correlated test patches. Not a patient-level or clinical benchmark.',
        fontsize=8.5,color='#526575')
    for ext in ('pdf','png','svg'):
        fig.savefig(out/f'original_tissue_cvae_spatial_error.{ext}',dpi=240,facecolor='white')
    plt.close(fig)
    print(json.dumps({'original_source_images':1,'heldout_patches':len(rows),
         'mean_test_PSNR_dB':float(psnr.mean()),'mean_test_SSIM':float(ssim.mean()),
         'mean_test_MSE':float(mse.mean()),'nearest_training_MSE':float(base.mean())},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--results',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    a=ap.parse_args();plot(a.results,a.output)
