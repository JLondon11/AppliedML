"""Evaluate zero-fill vs TV-projection reconstruction from REAL fastMRI raw HDF5.
Requires h5py, numpy, scikit-image, matplotlib. This is NOT a trained model.
"""
import argparse,csv,hashlib,json
from pathlib import Path
import h5py,numpy as np
from skimage.metrics import structural_similarity,peak_signal_noise_ratio
from skimage.restoration import denoise_tv_chambolle

def centered_fft2(a):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(a,axes=(-2,-1)),axes=(-2,-1),norm='ortho'),axes=(-2,-1))
def centered_ifft2(a):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(a,axes=(-2,-1)),axes=(-2,-1),norm='ortho'),axes=(-2,-1))
def cartesian_mask(shape,acceleration=4,center_fraction=.08):
    h,w=shape
    if acceleration<2 or not 0<center_fraction<1:raise ValueError('Invalid undersampling parameters')
    m=np.zeros(w,bool);m[::acceleration]=True
    n=int(round(w*center_fraction));start=(w-n)//2;m[start:start+n]=True
    return np.broadcast_to(m,(h,w)).copy()
def rss(coils):
    return np.sqrt(np.sum(abs(coils)**2,axis=0))
def pocs_tv(acquired,mask,iterations=12,weight=.014):
    est=centered_ifft2(acquired)
    for _ in range(iterations):
        den=np.stack([denoise_tv_chambolle(abs(coil),weight=weight) for coil in est])
        k=centered_fft2(den*np.exp(1j*np.angle(est)))
        k[:,mask]=acquired[:,mask]
        est=centered_ifft2(k)
    assert np.allclose(centered_fft2(est)[:,mask],acquired[:,mask],atol=1e-7,rtol=1e-7)
    return est
def assess(a,target):
    scale=max(float(target.max()),1e-12)
    x=np.asarray(a,float)/scale;y=np.asarray(target,float)/scale
    d=max(float(y.max()-y.min()),1e-12)
    return float(np.sqrt(np.mean((x-y)**2))),float(structural_similarity(x,y,data_range=d)),float(peak_signal_noise_ratio(y,x,data_range=d))
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()
def run_file(path,index,acceleration=4,center_fraction=.08):
    with h5py.File(path,'r') as f:
        if 'kspace' not in f:raise ValueError('Raw fastMRI kspace missing')
        source=f['kspace']
        if source.ndim not in (3,4):raise ValueError('Expected [slice,H,W] or [slice,coil,H,W]')
        raw=np.asarray(source[index])
    if raw.ndim==2:raw=raw[None]
    if not np.iscomplexobj(raw) or min(raw.shape[-2:])<32 or not np.isfinite(raw).all():
        raise ValueError('Real finite complex kspace required')
    mask=cartesian_mask(raw.shape[-2:],acceleration,center_fraction)
    measured=raw*mask[None]
    return dict(reference=rss(centered_ifft2(raw)),zero_fill=rss(centered_ifft2(measured)),
                tv_data_consistent=rss(pocs_tv(measured,mask)),mask=mask,
                mask_fraction=float(mask.mean()),shape=list(raw.shape))
def save_figures(result,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    images=[result['reference'],result['mask'],result['zero_fill'],result['tv_data_consistent'],
            abs(result['reference']-result['tv_data_consistent'])]
    fig,axs=plt.subplots(1,5,figsize=(13,3.4),facecolor='white')
    for i,(ax,img) in enumerate(zip(axs,images)):
        ax.imshow(img,cmap='magma' if i==4 else 'gray')
        ax.axis('off');ax.text(.5,-.07,f'({"abcde"[i]})',transform=ax.transAxes,ha='center',color='black')
    fig.subplots_adjust(left=.01,right=.99,bottom=.15,top=.98,wspace=.03)
    for ext in ('pdf','png','svg'):fig.savefig(out.with_suffix('.'+ext),dpi=200,bbox_inches='tight',facecolor='white')
    plt.close(fig)
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--h5',type=Path,required=True)
    p.add_argument('--slice-index',type=int,default=15)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--acceleration',type=int,default=4)
    p.add_argument('--center-fraction',type=float,default=.08)
    a=p.parse_args()
    if not a.h5.is_file():raise FileNotFoundError('Original raw fastMRI HDF5 required')
    r=run_file(a.h5,a.slice_index,a.acceleration,a.center_fraction)
    a.output.mkdir(parents=True,exist_ok=True)
    rows=[(k,*assess(r[k],r['reference'])) for k in ('zero_fill','tv_data_consistent')]
    with (a.output/'reconstruction_metrics.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['method','normalized_RMSE','SSIM','PSNR_dB']);w.writerows(rows)
    save_figures(r,a.output/'real_fastmri_reconstruction')
    manifest=dict(dataset='Original fastMRI HDF5',file_sha256=sha256(a.h5),slice_index=a.slice_index,
        requested_acceleration=a.acceleration,actual_sampling_fraction=r['mask_fraction'],shape=r['shape'],
        warning='Retrospective sampling, classical reconstruction only; NOT learned model or clinical scan-time evidence')
    (a.output/'experiment.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(rows)
if __name__=='__main__':main()
