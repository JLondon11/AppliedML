"""Real-data class-conditioned latent diffusion of sklearn handwritten digits.
Small class-conditioned VAE diffusion, NOT pretrained Stable Diffusion or text-to-image.
"""
from pathlib import Path
import argparse,csv,json,hashlib,random
import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SEED=20261109
class VAE(nn.Module):
 def __init__(self):
  super().__init__()
  self.encoder=nn.Sequential(nn.Linear(64,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU())
  self.mu=nn.Linear(64,16);self.lv=nn.Linear(64,16)
  self.decoder=nn.Sequential(nn.Linear(16,64),nn.SiLU(),nn.Linear(64,128),nn.SiLU(),nn.Linear(128,64),nn.Sigmoid())
 def encode(self,x):
  h=self.encoder(x);return self.mu(h),self.lv(h).clamp(-7,7)
 def forward(self,x):
  mu,lv=self.encode(x)
  z=mu+torch.randn_like(mu)*torch.exp(.5*lv)
  return self.decoder(z),mu,lv
class Denoiser(nn.Module):
 def __init__(self,conditional=True):
  super().__init__();self.conditional=conditional
  self.label=nn.Embedding(11,16)
  self.net=nn.Sequential(nn.Linear(48,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,16))
 def forward(self,x,t,y):
  if not self.conditional:y=torch.full_like(y,10)
  period=torch.exp(torch.linspace(0,6,8,device=x.device))
  angle=t.float()[:,None]/period[None,:]
  time=torch.cat([torch.sin(angle),torch.cos(angle)],1)
  return self.net(torch.cat([x,time,self.label(y)],1))
def seed_everything(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)
def diffusion_sample(net,alphas,abar,y,seed):
 gen=torch.Generator().manual_seed(seed);x=torch.randn((len(y),16),generator=gen)
 with torch.no_grad():
  for k in range(len(alphas)-1,-1,-1):
   t=torch.full((len(y),),k,dtype=torch.long)
   eps=net(x,t,y);a=alphas[k];ab=abar[k]
   mean=(x-(1-a)/torch.sqrt(1-ab)*eps)/torch.sqrt(a)
   if k>0:x=mean+torch.sqrt((1-ab/a)/(1-ab)*(1-a)).clamp(min=0)*torch.randn(x.shape,generator=gen)
   else:x=mean
 return x
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--vae-epochs',type=int,default=25)
 ap.add_argument('--diffusion-epochs',type=int,default=35)
 ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'results')
 args=ap.parse_args();out=args.output;figs=out/'figures';figs.mkdir(parents=True,exist_ok=True)
 seed_everything(SEED)
 source=load_digits();X=(source.data/16).astype('float32');Y=source.target.astype('int64')
 ids=np.arange(len(Y))
 train,other=train_test_split(ids,test_size=.3,random_state=SEED,stratify=Y)
 valid,test=train_test_split(other,test_size=.5,random_state=SEED,stratify=Y[other])
 assert len(set(train)&set(valid))==len(set(train)&set(test))==len(set(valid)&set(test))==0
 x=torch.from_numpy(X);y=torch.from_numpy(Y)
 torch.save({'train':train,'validation':valid,'test':test},out/'split_indices.pt')
 classifier=LogisticRegression(max_iter=2000,random_state=SEED).fit(X[train],Y[train])
 testacc=accuracy_score(Y[test],classifier.predict(X[test]))
 vae=VAE();opt=torch.optim.Adam(vae.parameters(),lr=.002)
 history=[];best=1e12;weights=None;bestepoch=0
 for epoch in range(1,args.vae_epochs+1):
  vae.train();perm=torch.tensor(np.random.permutation(train));l=[]
  for ix in perm.split(64):
   pred,mu,lv=vae(x[ix]);rec=(pred-x[ix]).square().mean()
   kl=(-.5*(1+lv-mu.square()-lv.exp()).sum(1)).mean()
   loss=rec+.0005*kl;opt.zero_grad();loss.backward();opt.step();l.append(float(loss.detach()))
  vae.eval()
  with torch.no_grad():
   pred,mu,lv=vae(x[valid]);score=float((pred-x[valid]).square().mean())
  history.append(['vae',epoch,float(np.mean(l)),score])
  if score<best:
   best=score;weights={k:v.detach().clone() for k,v in vae.state_dict().items()};bestepoch=epoch
 vae.load_state_dict(weights);vae.eval()
 with torch.no_grad():
  latent_train=vae.encode(x[train])[0].detach()
  testrecon=vae.decoder(vae.encode(x[test])[0]).numpy()
 scale=latent_train.std(0).clamp(min=.1);offset=latent_train.mean(0)
 ztrain=(latent_train-offset)/scale
 T=48;betas=torch.linspace(.0005,.055,T);alphas=1-betas;abar=torch.cumprod(alphas,0)
 records=[];generated_all={}
 for arm in ('conditional','unconditional'):
  seed_everything(SEED+(arm=='unconditional')*7)
  net=Denoiser(arm=='conditional');optimizer=torch.optim.Adam(net.parameters(),lr=.0015)
  for epoch in range(1,args.diffusion_epochs+1):
   net.train();order=torch.randperm(len(train));losses=[]
   for batch in order.split(64):
    z=ztrain[batch];label=y[train][batch];t=torch.randint(T,(len(batch),));noise=torch.randn_like(z)
    zt=torch.sqrt(abar[t])[:,None]*z+torch.sqrt(1-abar[t])[:,None]*noise
    loss=(net(zt,t,label)-noise).square().mean()
    optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
   history.append([arm,epoch,float(np.mean(losses)),''])
  net.eval()
  torch.save({'weights':net.state_dict(),'arm':arm,'steps':T,'epochs':args.diffusion_epochs},out/f'diffusion_{arm}.pt')
  target=torch.arange(10).repeat_interleave(100)
  zz=diffusion_sample(net,alphas,abar,target,SEED+200+(arm=='unconditional')*41)
  with torch.no_grad():generated=vae.decoder(zz*scale+offset).numpy()
  generated_all[arm]=generated
  probabilities=classifier.predict_proba(generated);predicted=probabilities.argmax(1)
  for i,(prompt,pred,probs) in enumerate(zip(target.numpy(),predicted,probabilities)):
   records.append([arm,i,int(prompt),int(pred),float(probs[prompt]),*map(float,probs)])
  distances=[]
  for part in np.array_split(generated,20):
   d=((part[:,None,:]-X[train][None,:,:])**2).mean(2)
   distances.extend(d.min(1).tolist())
  generated_all[arm+'_closest']=distances
  print(arm,'prompt_accuracy',float(np.mean(predicted==target.numpy())),flush=True)
 torch.save({'generated':generated_all['conditional'],'prompts':np.arange(10).repeat(100)},out/'generated_conditional.pt')
 torch.save({'generated':generated_all['unconditional'],'prompts':np.arange(10).repeat(100)},out/'generated_unconditional.pt')
 with (out/'predictions.csv').open('w',newline='') as f:
  writer=csv.writer(f);writer.writerow(['arm','index','prompt_label','classifier_prediction','prompt_confidence',*[f'p_{j}' for j in range(10)]]);writer.writerows(records)
 with (out/'training.csv').open('w',newline='') as f:
  writer=csv.writer(f);writer.writerow(['model','epoch','training_objective','validation_reconstruction_MSE']);writer.writerows(history)
 fig,axes=plt.subplots(3,10,figsize=(12,4),constrained_layout=True)
 for i in range(10):
  axes[0,i].imshow(generated_all['conditional'][i*100].reshape(8,8),vmin=0,vmax=1,cmap='gray')
  axes[1,i].imshow(generated_all['unconditional'][i*100].reshape(8,8),vmin=0,vmax=1,cmap='gray')
  ix=test[np.flatnonzero(Y[test]==i)[0]]
  axes[2,i].imshow(X[ix].reshape(8,8),vmin=0,vmax=1,cmap='gray')
  for r in range(3):axes[r,i].axis('off')
  axes[0,i].set_title(str(i),fontsize=11)
 fig.savefig(figs/'digit_conditioned_unconditioned_real.png',dpi=200);plt.close(fig)
 vals=[]
 for arm in ('conditional','unconditional'):
  r=[row for row in records if row[0]==arm]
  vals.append(float(np.mean([row[2]==row[3] for row in r])))
 fig,ax=plt.subplots(figsize=(6,4));ax.bar(['Class-conditioned latent diffusion','Unconditional latent diffusion'],vals)
 ax.set_ylim(0,1);ax.set_ylabel('Prompt-label consistency (real-trained classifier)')
 fig.tight_layout();fig.savefig(figs/'prompt_consistency.png',dpi=190);plt.close(fig)
 with torch.no_grad():recon_rmse=float(np.sqrt(((testrecon-X[test])**2).mean()))
 manifest={'source':'sklearn.load_digits() genuine handwritten-digit 8x8 scans',
 'source_sha256':hashlib.sha256(source.data.tobytes()+Y.tobytes()).hexdigest(),
 'n_total':len(ids),'n_train':len(train),'n_validation':len(valid),'n_test':len(test),
 'test_classifier_accuracy':float(testacc),'vae_selected_epoch':bestepoch,
 'vae_test_reconstruction_RMSE':recon_rmse,'diffusion_steps':T,
 'vae_epochs':args.vae_epochs,'diffusion_epochs':args.diffusion_epochs,'n_generated_per_arm':1000,
 'class_conditional_consistency':vals[0],'unconditional_consistency':vals[1],
 'conditional_nearest_train_MSE_mean':float(np.mean(generated_all['conditional_closest'])),
 'unconditional_nearest_train_MSE_mean':float(np.mean(generated_all['unconditional_closest'])),
 'scientific_scope':'small-class-label conditional VAE latent diffusion; NOT pretrained Stable Diffusion or text-to-image',
 'limitations':['single seed','coarse 8x8 grayscale digit images','class IDs are not natural language prompt embeddings',
                'classifier consistency is an imperfect realism measure','diffusion model convergence unproven','no FID/KID']}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
 torch.save({'weights':weights,'offset':offset,'scale':scale},out/'vae.pt')
 print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
