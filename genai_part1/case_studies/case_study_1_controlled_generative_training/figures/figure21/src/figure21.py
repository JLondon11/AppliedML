from pathlib import Path
import numpy as np,pandas as pd,torch,torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
BASE_SEED=21021; RUN_SEEDS=[21021,21022,21023,21024,21025]; EPOCHS=70; BETA=.12
OUT=Path(__file__).resolve().parent.parent/"outputs";OUT.mkdir(parents=True,exist_ok=True)
d=load_digits();X=(d.images.astype(np.float32)/16).reshape(len(d.images),-1);tr,va=train_test_split(np.arange(len(X)),test_size=.2,random_state=BASE_SEED,stratify=d.target);Xtr=torch.tensor(X[tr]);Xva=torch.tensor(X[va])
class VAE(nn.Module):
 def __init__(self):super().__init__();self.e=nn.Sequential(nn.Linear(64,96),nn.SiLU());self.mu=nn.Linear(96,12);self.lv=nn.Linear(96,12);self.dec=nn.Sequential(nn.Linear(12,96),nn.SiLU(),nn.Linear(96,64),nn.Sigmoid())
 def forward(self,x,st=True):
  h=self.e(x);m,l=self.mu(h),self.lv(h);z=m+torch.exp(.5*l)*torch.randn_like(m) if st else m;return self.dec(z),m,l
def terms(o,x,m,l):
 r=nn.functional.binary_cross_entropy(o,x,reduction="sum")/len(x);k=-.5*torch.sum(1+l-m.pow(2)-l.exp())/len(x);return r,k,r+BETA*k
rec=[]
for seed in RUN_SEEDS:
 torch.manual_seed(seed);np.random.seed(seed);loader=DataLoader(TensorDataset(Xtr),128,shuffle=True,generator=torch.Generator().manual_seed(seed));m=VAE();opt=torch.optim.Adam(m.parameters(),2e-3)
 for ep in range(1,EPOCHS+1):
  total=0;n=0
  for (xb,) in loader:
   o,mu,lv=m(xb);r,k,loss=terms(o,xb,mu,lv);opt.zero_grad();loss.backward();opt.step();total+=loss.item();n+=1
  with torch.no_grad():o,mu,lv=m(Xva,False);vr,vk,vl=terms(o,Xva,mu,lv)
  rec.append([seed,ep,total/n,vr.item(),vk.item(),vl.item()])
df=pd.DataFrame(rec,columns=["seed","epoch","train_objective","validation_reconstruction","validation_kl","validation_objective"]);df.to_csv(OUT/"figure21_training_runs.csv",index=False);g=df.groupby("epoch");s=pd.DataFrame({"epoch":range(1,EPOCHS+1),"train_mean":g.train_objective.mean().values,"train_sd":g.train_objective.std().values,"val_mean":g.validation_objective.mean().values,"val_sd":g.validation_objective.std().values,"rec_mean":g.validation_reconstruction.mean().values,"rec_sd":g.validation_reconstruction.std().values,"kl_mean":g.validation_kl.mean().values,"kl_sd":g.validation_kl.std().values});s.to_csv(OUT/"figure21_summary.csv",index=False)
navy="#173A52";violet="#725B78";olive="#66735B";fig,axs=plt.subplots(1,2,figsize=(7.4,3.65),dpi=240);x=s.epoch.values
axs[0].plot(x,s.train_mean,color=navy,lw=1.5,label="Training objective");axs[0].fill_between(x,s.train_mean-s.train_sd,s.train_mean+s.train_sd,color=navy,alpha=.14);axs[0].plot(x,s.val_mean,color=violet,lw=1.5,label="Validation objective");axs[0].fill_between(x,s.val_mean-s.val_sd,s.val_mean+s.val_sd,color=violet,alpha=.14);axs[0].set(xlabel="Epoch",ylabel="Negative ELBO objective");axs[0].legend(frameon=False,fontsize=8)
axs[1].plot(x,s.rec_mean,color=olive,lw=1.5,label="Reconstruction");axs[1].fill_between(x,s.rec_mean-s.rec_sd,s.rec_mean+s.rec_sd,color=olive,alpha=.14);a=axs[1].twinx();a.plot(x,s.kl_mean,color=violet,lw=1.35,label="KL");a.fill_between(x,s.kl_mean-s.kl_sd,s.kl_mean+s.kl_sd,color=violet,alpha=.12);axs[1].set(xlabel="Epoch",ylabel="Validation reconstruction term");a.set_ylabel("Validation KL term");h1,l1=axs[1].get_legend_handles_labels();h2,l2=a.get_legend_handles_labels();axs[1].legend(h1+h2,l1+l2,frameon=False,fontsize=8,loc="center right")
for ax in axs:ax.grid(True,lw=.4,alpha=.23)
fig.tight_layout();fig.savefig(OUT/"figure21.png",bbox_inches="tight");fig.savefig(OUT/"figure21.svg",bbox_inches="tight");plt.close(fig)
