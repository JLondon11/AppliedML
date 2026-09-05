from pathlib import Path
import numpy as np,pandas as pd,torch,torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
SEED=23023;np.random.seed(SEED);torch.manual_seed(SEED);OUT=Path(__file__).resolve().parent.parent/"outputs";OUT.mkdir(parents=True,exist_ok=True)
d=load_digits();X=(d.images.astype(np.float32)/16).reshape(-1,64);y=d.target.astype(np.int64);tr,te=train_test_split(np.arange(len(X)),test_size=.2,stratify=y,random_state=SEED);Xtr=torch.tensor(X[tr]);Xte=torch.tensor(X[te]);ytr=y[tr];yte=y[te];loader=DataLoader(TensorDataset(Xtr),128,shuffle=True,generator=torch.Generator().manual_seed(SEED))
class VAE(nn.Module):
 def __init__(self):super().__init__();self.h=nn.Sequential(nn.Linear(64,96),nn.SiLU());self.mu=nn.Linear(96,12);self.lv=nn.Linear(96,12);self.dec=nn.Sequential(nn.Linear(12,96),nn.SiLU(),nn.Linear(96,64),nn.Sigmoid())
 def encode(self,x):h=self.h(x);return self.mu(h),self.lv(h)
 def forward(self,x):m,l=self.encode(x);return self.dec(m+torch.exp(.5*l)*torch.randn_like(m)),m,l
m=VAE();o=torch.optim.Adam(m.parameters(),2e-3);losses=[]
for _ in range(85):
 total=0
 for (xb,) in loader:
  out,mu,lv=m(xb);loss=nn.functional.binary_cross_entropy(out,xb,reduction="sum")/len(xb)+.12*(-.5*torch.sum(1+lv-mu.pow(2)-lv.exp())/len(xb));o.zero_grad();loss.backward();o.step();total+=loss.item()
 losses.append(total/len(loader))
with torch.no_grad():ztr=m.encode(Xtr)[0].numpy();zte=m.encode(Xte)[0].numpy()
probe=LogisticRegression(max_iter=2500,random_state=SEED).fit(ztr,ytr);acc=accuracy_score(yte,probe.predict(zte));a=int(np.where(yte==1)[0][0]);b=int(np.where(yte==8)[0][0])
with torch.no_grad():
 za=m.encode(Xte[a:a+1])[0][0];zb=m.encode(Xte[b:b+1])[0][0];alpha=torch.linspace(0,1,11);zp=torch.stack([(1-t)*za+t*zb for t in alpha]);dec=m.dec(zp).view(-1,8,8).numpy();ls=torch.norm(zp[1:]-zp[:-1],dim=1).numpy();ims=torch.sqrt(torch.mean((m.dec(zp[1:])-m.dec(zp[:-1]))**2,dim=1)).numpy()
pd.DataFrame({"alpha":alpha.numpy(),"latent_norm_from_start":torch.norm(zp-za,dim=1).numpy()}).to_csv(OUT/"figure23_interpolation_path.csv",index=False);pd.DataFrame({"step":range(1,11),"latent_step_distance":ls,"decoded_image_rmse":ims}).to_csv(OUT/"figure23_step_metrics.csv",index=False);pd.DataFrame({"epoch":range(1,86),"vae_loss":losses}).to_csv(OUT/"figure23_training_log.csv",index=False);pd.DataFrame({"metric":["latent_probe_test_accuracy","source_class","target_class","mean_decoded_step_rmse"],"value":[acc,int(yte[a]),int(yte[b]),float(ims.mean())]}).to_csv(OUT/"figure23_summary.csv",index=False)
fig=plt.figure(figsize=(7.5,4.7),dpi=240);gs=fig.add_gridspec(2,11,height_ratios=[1,.92],hspace=.42,wspace=.06)
for j in range(11):
 ax=fig.add_subplot(gs[0,j]);ax.imshow(dec[j],cmap="gray",vmin=0,vmax=1,interpolation="nearest");ax.set_xticks([]);ax.set_yticks([]);ax.set_xlabel(f"{alpha[j]:.1f}",fontsize=7,labelpad=2)
ax=fig.add_subplot(gs[1,:]);navy="#173A52";violet="#725B78";ax.plot(range(1,11),ims,"o-",lw=1.45,ms=4,color=navy,label="Decoded-image RMSE");ax.set(xlabel="Interpolation step",ylabel="Decoded-image RMSE");ax.set_xticks(range(1,11));ax.grid(True,lw=.4,alpha=.23);ar=ax.twinx();ar.plot(range(1,11),ls,"s--",lw=1.15,ms=3.5,color=violet,label="Latent step distance");ar.set_ylabel("Latent step distance");h1,l1=ax.get_legend_handles_labels();h2,l2=ar.get_legend_handles_labels();ax.legend(h1+h2,l1+l2,frameon=False,fontsize=8,loc="upper center",ncol=2)
fig.savefig(OUT/"figure23.png",bbox_inches="tight");fig.savefig(OUT/"figure23.svg",bbox_inches="tight");plt.close(fig)
