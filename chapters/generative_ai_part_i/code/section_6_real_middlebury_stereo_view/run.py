"""Real Middlebury stereo-image conditional VAE. No synthetic driving benchmark.
One authentic stereo scene, spatially separated patches, actual opposite-camera evaluation.
"""
import argparse,csv,json,hashlib,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from skimage import data,metrics
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
  z=mu+torch.exp(.5*lv)*torch.randn_like(mu)
  return self.decode(left,z),mu,lv
def patchify(left,right,start,stop,patch=32,stride=16):
 xs=[];ys=[];coords=[]
 for row in range(start,stop-patch+1,stride):
  for col in range(0,left.shape[1]-patch+1,stride):
   xs.append(left[row:row+patch,col:col+patch].transpose(2,0,1));ys.append(right[row:row+patch,col:col+patch].transpose(2,0,1));coords.append((row,col))
 return torch.tensor(np.stack(xs),dtype=torch.float32),torch.tensor(np.stack(ys),dtype=torch.float32),coords
def make_metrics(left,right,gen):
 a=left.detach().numpy().transpose(0,2,3,1);b=right.detach().numpy().transpose(0,2,3,1);c=gen.detach().numpy().transpose(0,2,3,1)
 records=[]
 for i,(l,r,g) in enumerate(zip(a,b,c)):
  records.append({'test_patch':i,'PSNR_copy_left':float(metrics.peak_signal_noise_ratio(r,l,data_range=1)),
    'SSIM_copy_left':float(metrics.structural_similarity(r,l,channel_axis=2,data_range=1)),
    'PSNR_CVAE_prior_mean':float(metrics.peak_signal_noise_ratio(r,g,data_range=1)),
    'SSIM_CVAE_prior_mean':float(metrics.structural_similarity(r,g,channel_axis=2,data_range=1))})
 return records
def main():
 p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=220);p.add_argument('--batch',type=int,default=32)
 p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'results');args=p.parse_args()
 args.output.mkdir(parents=True,exist_ok=True);figs=args.output/'figures';figs.mkdir(exist_ok=True)
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(2)
 original_left,original_right,disp=data.stereo_motorcycle()
 left=original_left.astype('float32')/255;right=original_right.astype('float32')/255
 tr_l,tr_r,tr_c=patchify(left,right,0,320);va_l,va_r,va_c=patchify(left,right,320,400);te_l,te_r,te_c=patchify(left,right,400,500)
 assert (len(tr_l),len(va_l),len(te_l))==(855,180,225)
 model=CondVAE();opt=torch.optim.Adam(model.parameters(),lr=.001)
 best=1e9;selected=0;state=None;losses=[];rng=np.random.default_rng(SEED)
 for step in range(1,args.steps+1):
  model.train();idx=torch.tensor(rng.integers(0,len(tr_l),size=args.batch));inp=tr_l[idx];tar=tr_r[idx]
  pred,mu,lv=model(inp,tar);rec=(pred-tar).square().mean();kl=(-.5*(1+lv-mu.square()-lv.exp()).sum(1)).mean()
  loss=rec+.0005*kl
  opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),3);opt.step()
  if step%10==0 or step==args.steps:
   model.eval()
   with torch.no_grad():
    val=sum(float((model.decode(xx,torch.zeros(len(xx),model.zdim))-yy).square().mean())*len(xx)
            for xx,yy in zip(va_l.split(64),va_r.split(64)))/len(va_l)
   losses.append([step,float(loss.detach()),val])
   if val<best:
    best=val;selected=step;state={k:v.detach().clone() for k,v in model.state_dict().items()}
 model.load_state_dict(state);model.eval()
 pred=[];diverse=[]
 with torch.no_grad():
  for start in range(0,len(te_l),32):
   x=te_l[start:start+32];gen=torch.Generator().manual_seed(SEED+start+900)
   drawn=torch.stack([model.decode(x,torch.randn((len(x),model.zdim),generator=gen)) for j in range(8)])
   pred.append(drawn.mean(0));diverse.extend(drawn.std(0).mean((1,2,3)).tolist())
 pred=torch.cat(pred);records=make_metrics(te_l,te_r,pred)
 with (args.output/'test_metrics.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 with (args.output/'training.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['step','train_objective','validation_MSE']);w.writerows(losses)
 true=np.array([[r['PSNR_copy_left'],r['SSIM_copy_left'],r['PSNR_CVAE_prior_mean'],r['SSIM_CVAE_prior_mean']] for r in records]);means=true.mean(0)
 manifest={'dataset':'Middlebury 2014 motorcycle real rectified stereo photographs in scikit-image','source':'skimage.data.stereo_motorcycle()',
  'source_sha256_left':hashlib.sha256(original_left.tobytes()).hexdigest(),'source_sha256_right':hashlib.sha256(original_right.tobytes()).hexdigest(),
  'original_shape_left':list(original_left.shape),'original_shape_right':list(original_right.shape),
  'n_parent_stereo_pairs':1,'n_training_patches':len(tr_l),'n_validation_patches':len(va_l),'n_test_patches':len(te_l),
  'train_rows':[0,319],'validation_rows':[320,399],'test_rows':[400,499],
  'steps':args.steps,'selected_step':selected,'n_prior_draws_per_patch':8,'seed':SEED,
  'mean_test_PSNR_copy_left':means[0],'mean_test_SSIM_copy_left':means[1],
  'mean_test_PSNR_CVAE_prior_mean':means[2],'mean_test_SSIM_CVAE_prior_mean':means[3],
  'mean_generated_sample_spread':float(np.mean(diverse)),
  'scope':'conditional stochastic image translation on single real stereo scene; not synthetic autonomous-driving training data',
  'limitations':['Single parent stereo pair; correlated patches','No independent scene or traffic validation','Generated outputs are not real observations']}
 (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
 torch.save({'state':state,'manifest':manifest},args.output/'conditional_stereo_vae.pt')
 idxs=np.linspace(0,len(te_l)-1,6,dtype=int)
 f,axs=plt.subplots(4,6,figsize=(11.4,7.7),constrained_layout=True)
 for j,i in enumerate(idxs):
  for ax,im in zip(axs[:,j],[te_l[i].permute(1,2,0),te_r[i].permute(1,2,0),pred[i].permute(1,2,0),abs(pred[i]-te_r[i]).permute(1,2,0)*2]):
   ax.imshow(im.clamp(0,1));ax.set_axis_off()
  axs[0,j].set_title(f'Test patch {i}',fontsize=9)
 for row,label in enumerate(['Real left camera','Real right (target)','CVAE prior-predictive mean','Absolute error ×2']):
  axs[row,0].text(-.04,.5,label,va='center',ha='right',rotation=90,transform=axs[row,0].transAxes,fontsize=9,color='#244353')
 for ext in ('png','pdf'):f.savefig(figs/('stereo_real_model_results.'+ext),dpi=190,bbox_inches='tight')
 plt.close(f)
 f,axs=plt.subplots(1,2,figsize=(9.5,3.8),constrained_layout=True)
 for ax,x,y,label in zip(axs,[true[:,0],true[:,1]],[true[:,2],true[:,3]],['PSNR (dB)','SSIM']):
  ax.scatter(x,y,alpha=.38,s=17,color='#287483',edgecolor='white',linewidth=.2)
  lo=min(x.min(),y.min());hi=max(x.max(),y.max());ax.plot([lo,hi],[lo,hi],color='#B26A4A',ls='--',lw=1.4)
  ax.set(xlabel='Copy-left baseline',ylabel='Conditional VAE prior mean',title=f'Held-out {label} • n={len(x)}')
  ax.grid(alpha=.15)
 for ext in ('png','pdf'):f.savefig(figs/('stereo_paired_metrics.'+ext),dpi=200,bbox_inches='tight')
 plt.close(f)
 print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
