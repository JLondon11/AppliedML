"""Independent real Middlebury Motorcycle left-to-right CVAE benchmark.
The study is restricted to spatial holdout patches from ONE original real stereo pair.
"""
import csv, hashlib, json, random, argparse
from pathlib import Path
import numpy as np, torch
from torch import nn
from skimage import data, metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SEED=20261110
class CondVAE(nn.Module):
    def __init__(self,zdim=8):
        super().__init__()
        self.enc=nn.Sequential(nn.Conv2d(6,24,4,2,1),nn.SiLU(),nn.Conv2d(24,32,4,2,1),nn.SiLU(),nn.Conv2d(32,48,4,2,1),nn.SiLU(),nn.Flatten())
        self.mu=nn.Linear(48*4*4,zdim);self.lv=nn.Linear(48*4*4,zdim)
        self.dec=nn.Sequential(nn.Conv2d(3+zdim,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,3,3,padding=1),nn.Sigmoid())
        self.zdim=zdim
    def decode(self,left,z):
        return self.dec(torch.cat([left,z[:,:,None,None].expand(-1,-1,*left.shape[-2:])],axis=1))
    def forward(self,left,right):
        h=self.enc(torch.cat([left,right],1));mu=self.mu(h);lv=self.lv(h).clamp(-7,5)
        return self.decode(left,mu+torch.exp(.5*lv)*torch.randn_like(mu)),mu,lv

def patchify(left,right,start,stop,patch=32,stride=16):
    xs=[];ys=[];coords=[]
    for row in range(start,stop-patch+1,stride):
        for col in range(0,left.shape[1]-patch+1,stride):
            xs.append(left[row:row+patch,col:col+patch].transpose(2,0,1))
            ys.append(right[row:row+patch,col:col+patch].transpose(2,0,1));coords.append((row,col))
    return torch.tensor(np.stack(xs),dtype=torch.float32),torch.tensor(np.stack(ys),dtype=torch.float32),coords

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=220)
    ap.add_argument('--output',type=Path,default=Path('results'));args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(2)
    origL,origR,_=data.stereo_motorcycle()
    L=origL.astype('float32')/255;R=origR.astype('float32')/255
    tr_l,tr_r,trc=patchify(L,R,0,320);va_l,va_r,vac=patchify(L,R,320,400);te_l,te_r,tec=patchify(L,R,400,500)
    assert (len(tr_l),len(va_l),len(te_l))==(855,180,225)
    assert max(x for x,y in trc)+32<=min(x for x,y in vac)
    assert max(x for x,y in vac)+32<=min(x for x,y in tec)
    model=CondVAE();opt=torch.optim.Adam(model.parameters(),lr=.001)
    best=float('inf');chosen=0;state=None;history=[];rng=np.random.default_rng(SEED)
    for step in range(1,args.steps+1):
        model.train();idx=torch.tensor(rng.integers(0,len(tr_l),size=32)); inp=tr_l[idx];tar=tr_r[idx]
        pred,mu,lv=model(inp,tar);rec=(pred-tar).square().mean();kl=(-.5*(1+lv-mu.square()-lv.exp()).sum(1)).mean()
        loss=rec+.0005*kl
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),3);opt.step()
        if step%10==0 or step==args.steps:
            model.eval()
            with torch.no_grad():
                val=sum(float((model.decode(xx,torch.zeros(len(xx),model.zdim))-yy).square().mean())*len(xx) for xx,yy in zip(va_l.split(64),va_r.split(64)))/len(va_l)
            history.append((step,float(loss.detach()),val))
            if val<best:
                best=val;chosen=step;state={k:v.detach().clone() for k,v in model.state_dict().items()}
    assert state is not None
    model.load_state_dict(state);model.eval()
    preds=[];spread=[]
    with torch.no_grad():
        for start in range(0,len(te_l),32):
            x=te_l[start:start+32];gen=torch.Generator().manual_seed(SEED+start+900)
            draws=torch.stack([model.decode(x,torch.randn((len(x),model.zdim),generator=gen)) for _ in range(8)])
            preds.append(draws.mean(0));spread.extend(draws.std(0).mean((1,2,3)).tolist())
    preds=torch.cat(preds)
    a=te_l.numpy().transpose(0,2,3,1);b=te_r.numpy().transpose(0,2,3,1);c=preds.numpy().transpose(0,2,3,1)
    q=[]
    for i,(l,r,g) in enumerate(zip(a,b,c)):
        q.append(dict(test_patch=i,PSNR_copy_left=float(metrics.peak_signal_noise_ratio(r,l,data_range=1)),SSIM_copy_left=float(metrics.structural_similarity(r,l,channel_axis=2,data_range=1)),PSNR_CVAE_prior_mean=float(metrics.peak_signal_noise_ratio(r,g,data_range=1)),SSIM_CVAE_prior_mean=float(metrics.structural_similarity(r,g,channel_axis=2,data_range=1))))
    with (args.output/'test_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(q[0]));w.writeheader();w.writerows(q)
    with (args.output/'training.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['step','train_objective','validation_MSE']);w.writerows(history)
    z=np.array([[v[k] for k in ('PSNR_copy_left','SSIM_copy_left','PSNR_CVAE_prior_mean','SSIM_CVAE_prior_mean')] for v in q]);mean=z.mean(0)
    report=dict(dataset='Middlebury Motorcycle (actual original source photographs)',source='skimage.data.stereo_motorcycle()',left_sha256=hashlib.sha256(origL.tobytes()).hexdigest(),right_sha256=hashlib.sha256(origR.tobytes()).hexdigest(),parent_scenes=1,train=855,validation=180,test=225,steps=args.steps,seed=SEED,selected_step=chosen,psnr_copy=float(mean[0]),ssim_copy=float(mean[1]),psnr_cvae=float(mean[2]),ssim_cvae=float(mean[3]),psnr_cvae_win_patches=int((z[:,2]>z[:,0]).sum()),ssim_cvae_win_patches=int((z[:,3]>z[:,1]).sum()),mean_stochastic_spread=float(np.mean(spread)),scope='within-scene conditional right-view image generation; no driving downstream evaluation')
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    torch.save({'state':state,'manifest':report},args.output/'conditional_stereo_vae.pt')
    fig,ax=plt.subplots(1,2,figsize=(9.6,4.2),constrained_layout=True)
    for axis,x,y,label in zip(ax,[z[:,0],z[:,1]],[z[:,2],z[:,3]],['PSNR (dB)','SSIM']):
        axis.scatter(x,y,s=18,alpha=.48,color='#237987',edgecolor='white',linewidth=.22)
        lo=min(x.min(),y.min());hi=max(x.max(),y.max());axis.plot([lo,hi],[lo,hi],ls='--',lw=1.3,color='#AF735A')
        axis.set(xlabel='Copy-left baseline',ylabel='CVAE prior-predictive mean',title=f'{label}: 225 original held-out target patches');axis.grid(alpha=.18)
    fig.savefig(args.output/'paired_metrics.png',dpi=210,bbox_inches='tight');fig.savefig(args.output/'paired_metrics.svg',bbox_inches='tight');plt.close(fig)
    ids=np.linspace(0,len(te_l)-1,6,dtype=int)
    fig,ax=plt.subplots(4,6,figsize=(11,7.6),constrained_layout=True)
    for j,i in enumerate(ids):
        for axis,image in zip(ax[:,j],[te_l[i].permute(1,2,0),te_r[i].permute(1,2,0),preds[i].permute(1,2,0),abs(preds[i]-te_r[i]).permute(1,2,0)*2]):
            axis.imshow(image.clamp(0,1));axis.axis('off')
        ax[0,j].set_title(f'Test patch {i}',fontsize=9)
    for k,name in enumerate(['Real left','Real right target','Generated right mean','Absolute error x2']):
        ax[k,0].text(-.04,.5,name,ha='right',va='center',rotation=90,transform=ax[k,0].transAxes,fontsize=9)
    fig.savefig(args.output/'real_stereo_gallery.png',dpi=190,bbox_inches='tight');plt.close(fig)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
