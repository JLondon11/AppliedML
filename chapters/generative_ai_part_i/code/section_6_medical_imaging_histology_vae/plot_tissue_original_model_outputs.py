"""Image-first genuine held-out tissue model-output gallery from an executed checkpoint.

This script plots original measured photo patches, actual VAE reconstructed
pixels, absolute error and separately sampled model prior patches; NOT patients.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from skimage import data,metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class VAE(nn.Module):
    def __init__(self,z=24):
        super().__init__()
        self.enc=nn.Sequential(nn.Conv2d(3,16,4,2,1),nn.ReLU(),
            nn.Conv2d(16,32,4,2,1),nn.ReLU(),
            nn.Conv2d(32,64,4,2,1),nn.ReLU(),nn.Flatten())
        self.mu=nn.Linear(64*4*4,z);self.lv=nn.Linear(64*4*4,z)
        self.fc=nn.Linear(z,64*4*4)
        self.dec=nn.Sequential(nn.Unflatten(1,(64,4,4)),
            nn.ConvTranspose2d(64,32,4,2,1),nn.ReLU(),
            nn.ConvTranspose2d(32,16,4,2,1),nn.ReLU(),
            nn.ConvTranspose2d(16,3,4,2,1),nn.Sigmoid())
    def encode(self,x):
        h=self.enc(x);return self.mu(h),self.lv(h).clamp(-8,8)
    def decode(self,z):return self.dec(self.fc(z))

def main(checkpoint,out):
    torch.set_num_threads(2)
    src=data.immunohistochemistry()
    assert src.shape==(512,512,3)
    digest=hashlib.sha256(src.tobytes()).hexdigest()
    seq=[];pos=[]
    for row in range(384,512-31,16):
        for col in range(0,512-31,16):
            seq.append(np.moveaxis(src[row:row+32,col:col+32],-1,0))
            pos.append((row,col))
    assert len(seq)==217
    x=torch.from_numpy(np.stack(seq).astype('float32')/255)
    checkpoint_data=torch.load(checkpoint,map_location='cpu',weights_only=True)
    net=VAE()
    net.load_state_dict(checkpoint_data['weights'] if 'weights' in checkpoint_data
                        else checkpoint_data['model'])
    net.eval()
    with torch.no_grad():
        predicted=torch.cat([net.decode(net.encode(batch)[0]) for batch in x.split(64)])
        generator=torch.Generator().manual_seed(20261109+111)
        prior=net.decode(torch.randn(8,24,generator=generator))
    actual=np.moveaxis(x.numpy(),1,-1)
    reconstructed=np.moveaxis(predicted.numpy(),1,-1)
    sampled=np.moveaxis(prior.numpy(),1,-1)
    psnr=np.array([metrics.peak_signal_noise_ratio(a,b,data_range=1)
                   for a,b in zip(actual,reconstructed)])
    ssim=np.array([metrics.structural_similarity(a,b,channel_axis=-1,data_range=1)
                   for a,b in zip(actual,reconstructed)])
    mse=((actual-reconstructed)**2).mean(axis=(1,2,3))
    if not np.isfinite(np.r_[psnr,ssim,mse]).all():
        raise ValueError('Invalid real tissue model error measurements')
    indices=[2,31,59,93,125,158,186,213]
    fig,axes=plt.subplots(4,8,figsize=(12.6,7.2),facecolor='white')
    for column,i in enumerate(indices):
        err=np.abs(actual[i]-reconstructed[i]).mean(axis=2)
        for r,arr,cmap in ((0,actual[i],None),(1,reconstructed[i],None),
                           (2,err,'magma'),(3,sampled[column],None)):
            a=axes[r,column]
            a.imshow(arr,cmap=cmap,vmin=0,vmax=.5 if r==2 else 1,
                     interpolation='nearest')
            a.set_xticks([]);a.set_yticks([])
        axes[0,column].set_title(f'Patch {i}\n{pos[i]}',fontsize=8)
        axes[1,column].text(.02,.02,f'PSNR {psnr[i]:.1f} dB',
            transform=axes[1,column].transAxes,color='white',fontsize=6.5,
            bbox={'facecolor':'#173e5b','alpha':.86,'edgecolor':'none','pad':1})
    labels=('Original held-out\ntissue patch','Trained VAE\nreconstruction',
            'Absolute RGB\npixel error','VAE generated\nprior sample')
    for r,label in enumerate(labels):
        fig.text(.018,.76-.202*r,label,fontsize=8.6,ha='left',va='center',color='#214962')
    fig.suptitle('Original tissue photo → trained VAE reconstruction → measured error',
        x=.035,y=.975,ha='left',fontsize=16,weight='bold',color='#173e5b')
    fig.text(.035,.026,'One source photograph; 217 correlated held-out patches, not independent patients. Error color scale 0–0.5.',
        fontsize=8.6,color='#536a79')
    fig.subplots_adjust(left=.18,right=.99,top=.87,bottom=.10,hspace=.26,wspace=.09)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for ext in ('pdf','png','svg'):
        fig.savefig(out/f'actual_tissue_cvae_reconstruction_gallery.{ext}',
            dpi=200,facecolor='white')
    plt.close(fig)
    report={'original_source_sha256':digest,'source_images':1,
        'spatially_heldout_patches':217,
        'actual_model_test_PSNR_dB':float(psnr.mean()),
        'actual_model_test_SSIM':float(ssim.mean()),
        'actual_model_test_MSE':float(mse.mean()),
        'selected_original_test_indices':indices,
        'source_is_not_multiple_patients':True,
        'note':'True original-source model predictions; no clinical or cross-image generalization'}
    (out/'actual_tissue_gallery_measured_metrics.json').write_text(
        json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();main(args.checkpoint,args.output)
