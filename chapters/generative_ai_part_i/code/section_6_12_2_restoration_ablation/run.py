"""Chapter 6 restoration ablation on genuine scikit-image photographs.

Three seeded, single-step epsilon-prediction models. Not full DDPM sampling.
Execute: python run.py --steps 65
"""
from __future__ import annotations
import csv,json,random,hashlib,argparse
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import data,color,transform,metrics
import skimage
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEEDS=(20261106,20261107,20261108)
TRAIN=('astronaut','coffee','chelsea','rocket')
TEST=('camera','coins')
ALPHAS=(.90,.75)

def gray(im):
    if im.ndim==3:im=color.rgb2gray(im[:,:,:3])
    if np.issubdtype(im.dtype,np.integer):im=im.astype(np.float32)/np.iinfo(im.dtype).max
    return transform.resize(np.asarray(im,dtype=np.float32),(128,128),anti_aliasing=True).astype(np.float32)

class EpsCNN(nn.Module):
    def __init__(self,conditioned=True):
        super().__init__();self.conditioned=conditioned
        self.net=nn.Sequential(nn.Conv2d(2,32,3,padding=1),nn.SiLU(),
            nn.Conv2d(32,32,3,padding=1),nn.SiLU(),
            nn.Conv2d(32,32,3,padding=1),nn.SiLU(),
            nn.Conv2d(32,1,3,padding=1))
    def forward(self,x,alpha):
        conditioning=alpha[:,None,None,None].expand_as(x) if self.conditioned else torch.zeros_like(x)
        return self.net(torch.cat((x,conditioning),dim=1))

def train_one(seed,conditioned,training,steps,out):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)
    model=EpsCNN(conditioned);opt=torch.optim.Adam(model.parameters(),lr=.002)
    rng=np.random.default_rng(seed);losses=[]
    for step in range(steps):
        inds=rng.integers(0,len(training),size=24)
        xs=rng.integers(0,97,size=24);ys=rng.integers(0,97,size=24)
        clean=np.stack([training[i,y:y+32,x:x+32] for i,x,y in zip(inds,xs,ys)])[:,None]
        clean=torch.from_numpy(clean.copy());noise=torch.randn_like(clean)
        alpha=torch.from_numpy(rng.uniform(.70,.98,size=24).astype('float32'))
        xt=alpha[:,None,None,None].sqrt()*clean+(1-alpha[:,None,None,None]).sqrt()*noise
        loss=(model(xt,alpha)-noise).square().mean()
        opt.zero_grad();loss.backward();opt.step()
        losses.append((seed,'conditioned' if conditioned else 'unconditioned',step+1,float(loss.detach())))
    label='conditioned' if conditioned else 'unconditioned'
    torch.save({'model':model.state_dict(),'seed':seed,'conditioned':conditioned,'steps':steps},
               out/f'model_{label}_seed{seed}.pt')
    model.eval()
    return model,losses

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--steps',type=int,default=65)
    ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'results')
    a=ap.parse_args();out=a.output;figs=out/'figures';figs.mkdir(parents=True,exist_ok=True)
    originals={nm:gray(getattr(data,nm)()) for nm in (*TRAIN,*TEST)}
    training=np.stack([originals[k] for k in TRAIN]);held={k:originals[k] for k in TEST}
    rows=[];history=[];plot_images={}
    noisy_cases={}
    for image_name,ref in held.items():
        for alpha in ALPHAS:
            noise=np.random.default_rng(20261106+len(image_name)*100+int(alpha*100)).standard_normal(ref.shape).astype('float32')
            xt=np.sqrt(alpha)*ref+np.sqrt(1-alpha)*noise
            noisy_cases[(image_name,alpha)]=xt.astype('float32')
    for seed in SEEDS:
        models={}
        for conditioned in (False,True):
            name='conditioned' if conditioned else 'unconditioned'
            model,h=train_one(seed,conditioned,training,a.steps,out);history+=h;models[name]=model
        for (imname,alpha),xt in noisy_cases.items():
            ref=held[imname];inp=np.clip(xt/np.sqrt(alpha),0,1)
            smooth=np.clip(gaussian_filter(inp,sigma=.8),0,1)
            variants={'noisy':inp,'gaussian_sigma_0.8':smooth}
            for label,model in models.items():
                with torch.no_grad():
                    epshat=model(torch.from_numpy(xt[None,None]),
                                 torch.tensor([alpha],dtype=torch.float32)).numpy()[0,0]
                variants[label]=np.clip((xt-np.sqrt(1-alpha)*epshat)/np.sqrt(alpha),0,1)
            for name,z in variants.items():
                rows.append({'seed':seed,'image':imname,'alpha_bar':alpha,'model':name,
                    'psnr_db':float(metrics.peak_signal_noise_ratio(ref,z,data_range=1)),
                    'ssim':float(metrics.structural_similarity(ref,z,data_range=1))})
            if seed==SEEDS[0] and alpha==.75:
                plot_images[imname]=(ref,inp,smooth,variants['unconditioned'],variants['conditioned'])
    with (out/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (out/'training.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['seed','model','step','epsilon_mse']);w.writerows(history)
    import pandas as pd
    frame=pd.DataFrame(rows)
    means=frame.groupby('model').agg(psnr_mean=('psnr_db','mean'),psnr_sd=('psnr_db','std'),
        ssim_mean=('ssim','mean'),ssim_sd=('ssim','std')).reset_index()
    means.to_csv(out/'summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(8.1,4.1))
    labels=['noisy','gaussian_sigma_0.8','unconditioned','conditioned']
    for i,k in enumerate(labels):
        vals=frame[frame.model==k].psnr_db.to_numpy()
        ax.bar(i,float(vals.mean()),width=.66)
        ax.errorbar(i,vals.mean(),yerr=vals.std(ddof=1),fmt='none',ecolor='black',capsize=4)
    ax.set_xticks(range(4),['Corrupted','Gaussian','No time input','Time conditioned'])
    ax.set_ylabel('Held-out PSNR (dB)')
    ax.set_ylim(0,max(frame.psnr_db)*1.18);fig.tight_layout()
    fig.savefig(figs/'ablation_psnr.png',dpi=190);plt.close(fig)
    fig,axs=plt.subplots(2,5,figsize=(11,5.2),constrained_layout=True)
    for i,nm in enumerate(TEST):
        for j,z in enumerate(plot_images[nm]):
            axs[i,j].imshow(z,cmap='gray',vmin=0,vmax=1);axs[i,j].axis('off')
            if i==0:
                axs[i,j].set_title(['Reference','Corrupted','Gaussian','No time input','Time conditioned'][j],fontsize=9)
            if j==0:
                axs[i,j].text(.02,.96,nm,va='top',color='white',
                  transform=axs[i,j].transAxes,bbox={'facecolor':'black','alpha':.65,'pad':2})
    fig.savefig(figs/'real_photo_ablation_outputs.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3.6))
    for label in ('conditioned','unconditioned'):
        curves=np.stack([np.array([r[3] for r in history if r[0]==seed and r[1]==label]) for seed in SEEDS])
        ax.plot(np.arange(1,a.steps+1),curves.mean(0),label=label)
    ax.set(xlabel='Optimizer step',ylabel='Training epsilon MSE')
    ax.legend(frameon=False);fig.tight_layout()
    fig.savefig(figs/'ablation_training.png',dpi=180);plt.close(fig)
    manifest={'purpose':'single-step epsilon prediction ablation, not complete diffusion sampler',
      'image_source':'scikit-image shipped photograph collection',
      'training_images':list(TRAIN),'heldout_test_images':list(TEST),
      'seeds':list(SEEDS),'steps_per_model':a.steps,
      'train_images':4,'test_images':2,'alpha_bar':list(ALPHAS),
      'noise_protocol':'same fixed Gaussian corruption for each arm and seed',
      'baseline':'Gaussian smoothing sigma 0.8 (not tuned to held-out data)',
      'limitations':['only two held-out source photographs','corruption simulated on real photographs',
       'no validation split','not scientific microscopy or full iterative reverse diffusion'],
      'versions':{'torch':torch.__version__,'skimage':skimage.__version__,'numpy':np.__version__},
      'sha256':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in originals.items()}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(means.to_string(index=False))

if __name__=='__main__':main()
