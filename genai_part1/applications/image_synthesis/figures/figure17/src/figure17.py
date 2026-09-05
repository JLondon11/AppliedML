from pathlib import Path
import numpy as np,pandas as pd,torch,torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
SEED=17017; torch.manual_seed(SEED); np.random.seed(SEED)
OUT=Path(__file__).resolve().parent.parent/"outputs"; OUT.mkdir(parents=True,exist_ok=True)
d=load_digits(); X=torch.tensor(d.images/16.,dtype=torch.float32).view(-1,64)
loader=DataLoader(TensorDataset(X),128,shuffle=True,generator=torch.Generator().manual_seed(SEED))
class VAE(nn.Module):
 def __init__(self):
  super().__init__(); self.e=nn.Sequential(nn.Linear(64,96),nn.SiLU()); self.mu=nn.Linear(96,12); self.lv=nn.Linear(96,12); self.d=nn.Sequential(nn.Linear(12,96),nn.SiLU(),nn.Linear(96,64),nn.Sigmoid())
 def forward(self,x):
  h=self.e(x); m,l=self.mu(h),self.lv(h); z=m+torch.exp(.5*l)*torch.randn_like(m); return self.d(z),m,l
vae=VAE(); opt=torch.optim.Adam(vae.parameters(),2e-3); vl=[]
for _ in range(45):
 t=0
 for (x,) in loader:
  y,m,l=vae(x); loss=nn.functional.binary_cross_entropy(y,x,reduction="sum")/len(x)+.15*(-.5*torch.sum(1+l-m.pow(2)-l.exp())/len(x)); opt.zero_grad(); loss.backward(); opt.step(); t+=loss.item()
 vl.append(t/len(loader))
class G(nn.Module):
 def __init__(self): super().__init__(); self.n=nn.Sequential(nn.Linear(24,96),nn.LeakyReLU(.2),nn.Linear(96,128),nn.LeakyReLU(.2),nn.Linear(128,64),nn.Sigmoid())
 def forward(self,z): return self.n(z)
class D(nn.Module):
 def __init__(self): super().__init__(); self.n=nn.Sequential(nn.Linear(64,128),nn.LeakyReLU(.2),nn.Linear(128,64),nn.LeakyReLU(.2),nn.Linear(64,1))
 def forward(self,x): return self.n(x)
g,disc=G(),D(); og=torch.optim.Adam(g.parameters(),2e-4,betas=(.5,.999)); od=torch.optim.Adam(disc.parameters(),2e-4,betas=(.5,.999)); bce=nn.BCEWithLogitsLoss(); gl=[]
for _ in range(80):
 t=0
 for (x,) in loader:
  bs=len(x); fake=g(torch.randn(bs,24)); ld=bce(disc(x),torch.ones(bs,1))+bce(disc(fake.detach()),torch.zeros(bs,1)); od.zero_grad(); ld.backward(); od.step(); lg=bce(disc(g(torch.randn(bs,24))),torch.ones(bs,1)); og.zero_grad(); lg.backward(); og.step(); t+=lg.item()
 gl.append(t/len(loader))
T=80; betas=torch.linspace(1e-4,.08,T); alphas=1-betas; abar=torch.cumprod(alphas,0)
class Eps(nn.Module):
 def __init__(self): super().__init__(); self.n=nn.Sequential(nn.Linear(65,192),nn.SiLU(),nn.Linear(192,192),nn.SiLU(),nn.Linear(192,64))
 def forward(self,x,t): return self.n(torch.cat([x,(t.float()/(T-1)).unsqueeze(1)],1))
eps=Eps(); oe=torch.optim.Adam(eps.parameters(),1e-3); dl=[]
for _ in range(70):
 total=0
 for (x0,) in loader:
  bs=len(x0); ti=torch.randint(0,T,(bs,)); noise=torch.randn_like(x0); ab=abar[ti].unsqueeze(1); xt=ab.sqrt()*x0+(1-ab).sqrt()*noise; loss=nn.functional.mse_loss(eps(xt,ti),noise); oe.zero_grad(); loss.backward(); oe.step(); total+=loss.item()
 dl.append(total/len(loader))
with torch.no_grad():
 real=X[:8].view(-1,8,8).numpy(); vs=vae.d(torch.randn(8,12)).view(-1,8,8).numpy(); gs=g(torch.randn(8,24)).view(-1,8,8).numpy(); x=torch.randn(8,64)
 for ti in reversed(range(T)):
  tt=torch.full((8,),ti,dtype=torch.long); a=alphas[ti]; ab=abar[ti]; b=betas[ti]; mean=(x-(b/torch.sqrt(1-ab))*eps(x,tt))/torch.sqrt(a)
  x=mean+(torch.sqrt(b*(1-abar[ti-1])/(1-ab))*torch.randn_like(x) if ti>0 else 0)
 ds=x.clamp(0,1).view(-1,8,8).numpy()
np.savez_compressed(OUT/"figure17_samples.npz",observed=real,vae=vs,gan=gs,ddpm=ds)
pd.DataFrame({"vae_loss":vl+[np.nan]*35,"gan_generator_loss":gl,"ddpm_noise_mse":dl+[np.nan]*10}).to_csv(OUT/"figure17_training_log.csv",index=False)
fig,axs=plt.subplots(4,8,figsize=(7.4,3.9),dpi=240)
for i,(arr,label) in enumerate(zip([real,vs,gs,ds],["Observed","VAE","GAN","DDPM"])):
 for j in range(8):
  axs[i,j].imshow(arr[j],cmap="gray",vmin=0,vmax=1,interpolation="nearest"); axs[i,j].set_xticks([]); axs[i,j].set_yticks([])
 axs[i,0].set_ylabel(label,rotation=0,ha="right",va="center",labelpad=13,fontsize=9)
plt.subplots_adjust(wspace=.06,hspace=.12,left=.09,right=.995,top=.99,bottom=.02)
fig.savefig(OUT/"figure17.png",bbox_inches="tight"); fig.savefig(OUT/"figure17.svg",bbox_inches="tight"); plt.close(fig)
