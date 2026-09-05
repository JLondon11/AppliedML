"""Figure 17 — Image Synthesis Across Generative Model Families.

Actual executable experiment on sklearn's public 8x8 handwritten-digits dataset.
Trains a VAE, GAN, and DDPM from scratch under a fixed seed, then renders authentic
dataset observations and actual model outputs. No image-generation service is used.
"""
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

SEED=17017; torch.manual_seed(SEED); np.random.seed(SEED)
digits=load_digits()
X=torch.tensor(digits.images/16.0,dtype=torch.float32).view(-1,64)
loader=DataLoader(TensorDataset(X),batch_size=128,shuffle=True,
 generator=torch.Generator().manual_seed(SEED))

class VAE(nn.Module):
 def __init__(self):
  super().__init__(); self.e=nn.Sequential(nn.Linear(64,96),nn.SiLU())
  self.mu=nn.Linear(96,12); self.lv=nn.Linear(96,12)
  self.d=nn.Sequential(nn.Linear(12,96),nn.SiLU(),nn.Linear(96,64),nn.Sigmoid())
 def forward(self,x):
  h=self.e(x); mu,lv=self.mu(h),self.lv(h)
  z=mu+torch.exp(.5*lv)*torch.randn_like(mu)
  return self.d(z),mu,lv

vae=VAE(); opt=torch.optim.Adam(vae.parameters(),2e-3); vae_losses=[]
for ep in range(45):
 tot=0
 for (x,) in loader:
  y,mu,lv=vae(x); rec=nn.functional.binary_cross_entropy(y,x,reduction="sum")/len(x)
  kl=-.5*torch.sum(1+lv-mu.pow(2)-lv.exp())/len(x); loss=rec+.15*kl
  opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item()
 vae_losses.append(tot/len(loader))

class G(nn.Module):
 def __init__(self):
  super().__init__(); self.n=nn.Sequential(nn.Linear(24,96),nn.LeakyReLU(.2),nn.Linear(96,128),nn.LeakyReLU(.2),nn.Linear(128,64),nn.Sigmoid())
 def forward(self,z): return self.n(z)
class D(nn.Module):
 def __init__(self):
  super().__init__(); self.n=nn.Sequential(nn.Linear(64,128),nn.LeakyReLU(.2),nn.Linear(128,64),nn.LeakyReLU(.2),nn.Linear(64,1))
 def forward(self,x): return self.n(x)

g,d=G(),D(); og=torch.optim.Adam(g.parameters(),2e-4,betas=(.5,.999)); od=torch.optim.Adam(d.parameters(),2e-4,betas=(.5,.999))
bce=nn.BCEWithLogitsLoss(); gan_losses=[]
for ep in range(80):
 gl=0
 for (x,) in loader:
  bs=len(x); fake=g(torch.randn(bs,24))
  ld=bce(d(x),torch.ones(bs,1))+bce(d(fake.detach()),torch.zeros(bs,1))
  od.zero_grad(); ld.backward(); od.step()
  lg=bce(d(g(torch.randn(bs,24))),torch.ones(bs,1))
  og.zero_grad(); lg.backward(); og.step(); gl+=lg.item()
 gan_losses.append(gl/len(loader))

T=80; betas=torch.linspace(1e-4,.08,T); alphas=1-betas; abar=torch.cumprod(alphas,0)
class Eps(nn.Module):
 def __init__(self):
  super().__init__(); self.n=nn.Sequential(nn.Linear(65,192),nn.SiLU(),nn.Linear(192,192),nn.SiLU(),nn.Linear(192,64))
 def forward(self,x,t): return self.n(torch.cat([x,(t.float()/(T-1)).unsqueeze(1)],1))
eps=Eps(); oe=torch.optim.Adam(eps.parameters(),1e-3); ddpm_losses=[]
for ep in range(70):
 tot=0
 for (x0,) in loader:
  bs=len(x0); t=torch.randint(0,T,(bs,)); noise=torch.randn_like(x0); ab=abar[t].unsqueeze(1)
  xt=ab.sqrt()*x0+(1-ab).sqrt()*noise; loss=nn.functional.mse_loss(eps(xt,t),noise)
  oe.zero_grad(); loss.backward(); oe.step(); tot+=loss.item()
 ddpm_losses.append(tot/len(loader))

with torch.no_grad():
 real=X[:8].view(-1,8,8).numpy(); vs=vae.d(torch.randn(8,12)).view(-1,8,8).numpy()
 gs=g(torch.randn(8,24)).view(-1,8,8).numpy(); x=torch.randn(8,64)
 for ti in reversed(range(T)):
  t=torch.full((8,),ti,dtype=torch.long); a=alphas[ti]; ab=abar[ti]; b=betas[ti]
  mean=(x-(b/torch.sqrt(1-ab))*eps(x,t))/torch.sqrt(a)
  if ti>0:
   prev=abar[ti-1]; x=mean+torch.sqrt(b*(1-prev)/(1-ab))*torch.randn_like(x)
  else: x=mean
 ds=x.clamp(0,1).view(-1,8,8).numpy()

out=Path(__file__).resolve().parent/"outputs"; out.mkdir(parents=True,exist_ok=True)
np.savez_compressed(out/"figure17_samples.npz",observed=real,vae=vs,gan=gs,ddpm=ds)
pd.DataFrame({"vae_loss":vae_losses+[np.nan]*35,"gan_generator_loss":gan_losses,
 "ddpm_noise_mse":ddpm_losses+[np.nan]*10}).to_csv(out/"figure17_training_log.csv",index=False)

fig,axs=plt.subplots(4,8,figsize=(7.4,3.9),dpi=240)
for i,(arr,label) in enumerate(zip([real,vs,gs,ds],["Observed","VAE","GAN","DDPM"])):
 for j in range(8):
  axs[i,j].imshow(arr[j],cmap="gray",vmin=0,vmax=1,interpolation="nearest")
  axs[i,j].set_xticks([]); axs[i,j].set_yticks([])
  for s in axs[i,j].spines.values(): s.set_linewidth(.35)
 axs[i,0].set_ylabel(label,rotation=0,ha="right",va="center",labelpad=13,fontsize=9)
plt.subplots_adjust(wspace=.06,hspace=.12,left=.09,right=.995,top=.99,bottom=.02)
fig.savefig(out/"figure17_image_synthesis.png",bbox_inches="tight")
fig.savefig(out/"figure17_image_synthesis.pdf",bbox_inches="tight")
