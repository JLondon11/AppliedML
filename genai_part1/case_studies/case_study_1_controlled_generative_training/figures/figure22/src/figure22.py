# Full Figure 22 experiment. See README for protocol.
from pathlib import Path
import time,numpy as np,pandas as pd,torch,torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
BASE=22022;SEEDS=[22022,22023,22024];EPOCHS=45;OUT=Path(__file__).resolve().parent.parent/"outputs";OUT.mkdir(parents=True,exist_ok=True)
np.random.seed(BASE);torch.manual_seed(BASE);ds=load_digits();X=(ds.images.astype(np.float32)/16).reshape(-1,64);tr,te=train_test_split(np.arange(len(X)),test_size=.2,stratify=ds.target,random_state=BASE);Xtr=torch.tensor(X[tr]);Xte=X[te]
nnr=NearestNeighbors(n_neighbors=2).fit(X[tr]);d,_=nnr.kneighbors(X[tr]);tau=float(np.quantile(d[:,1],.95))
def pr(gen):
 a=NearestNeighbors(n_neighbors=1).fit(X[tr]);dg=a.kneighbors(gen,return_distance=True)[0][:,0];b=NearestNeighbors(n_neighbors=1).fit(gen);dr=b.kneighbors(Xte,return_distance=True)[0][:,0];return float(np.mean(dg<=tau)),float(np.mean(dr<=tau))
class AE(nn.Module):
 def __init__(self):super().__init__();self.e=nn.Sequential(nn.Linear(64,96),nn.SiLU(),nn.Linear(96,12));self.d=nn.Sequential(nn.Linear(12,96),nn.SiLU(),nn.Linear(96,64),nn.Sigmoid())
 def forward(self,x):return self.d(self.e(x))
class VAE(nn.Module):
 def __init__(self):super().__init__();self.h=nn.Sequential(nn.Linear(64,96),nn.SiLU());self.mu=nn.Linear(96,12);self.lv=nn.Linear(96,12);self.d=nn.Sequential(nn.Linear(12,96),nn.SiLU(),nn.Linear(96,64),nn.Sigmoid())
 def forward(self,x):h=self.h(x);m,l=self.mu(h),self.lv(h);return self.d(m+torch.exp(.5*l)*torch.randn_like(m)),m,l
class G(nn.Module):
 def __init__(self):super().__init__();self.n=nn.Sequential(nn.Linear(24,96),nn.LeakyReLU(.2),nn.Linear(96,128),nn.LeakyReLU(.2),nn.Linear(128,64),nn.Sigmoid())
 def forward(self,z):return self.n(z)
class D(nn.Module):
 def __init__(self):super().__init__();self.n=nn.Sequential(nn.Linear(64,128),nn.LeakyReLU(.2),nn.Linear(128,64),nn.LeakyReLU(.2),nn.Linear(64,1))
 def forward(self,x):return self.n(x)
rows=[]
for seed in SEEDS:
 torch.manual_seed(seed);np.random.seed(seed);loader=DataLoader(TensorDataset(Xtr),128,shuffle=True,generator=torch.Generator().manual_seed(seed))
 ae=AE();o=torch.optim.Adam(ae.parameters(),2e-3);t=time.perf_counter()
 for _ in range(EPOCHS):
  for (x,) in loader:y=ae(x);loss=nn.functional.mse_loss(y,x);o.zero_grad();loss.backward();o.step()
 sec=time.perf_counter()-t
 with torch.no_grad():
  z=ae.e(Xtr);zm=z.mean(0);zc=z-zm;L=torch.linalg.cholesky((zc.T@zc)/(len(z)-1)+1e-4*torch.eye(12));ts=time.perf_counter();gen=ae.d(zm+torch.randn(1200,12)@L.T).numpy();ms=(time.perf_counter()-ts)/1.2
 p,r=pr(gen);rows.append([seed,"Latent AE",p,r,sec,ms,sum(q.numel() for q in ae.parameters())])
 vae=VAE();o=torch.optim.Adam(vae.parameters(),2e-3);t=time.perf_counter()
 for _ in range(EPOCHS):
  for (x,) in loader:y,m,l=vae(x);loss=nn.functional.binary_cross_entropy(y,x,reduction="sum")/len(x)+.12*(-.5*torch.sum(1+l-m.pow(2)-l.exp())/len(x));o.zero_grad();loss.backward();o.step()
 sec=time.perf_counter()-t
 with torch.no_grad():ts=time.perf_counter();gen=vae.d(torch.randn(1200,12)).numpy();ms=(time.perf_counter()-ts)/1.2
 p,r=pr(gen);rows.append([seed,"VAE",p,r,sec,ms,sum(q.numel() for q in vae.parameters())])
 g,disc=G(),D();og=torch.optim.Adam(g.parameters(),2e-4,betas=(.5,.999));od=torch.optim.Adam(disc.parameters(),2e-4,betas=(.5,.999));bce=nn.BCEWithLogitsLoss();t=time.perf_counter()
 for _ in range(EPOCHS):
  for (x,) in loader:
   bs=len(x);fake=g(torch.randn(bs,24));ld=bce(disc(x),torch.ones(bs,1))+bce(disc(fake.detach()),torch.zeros(bs,1));od.zero_grad();ld.backward();od.step();lg=bce(disc(g(torch.randn(bs,24))),torch.ones(bs,1));og.zero_grad();lg.backward();og.step()
 sec=time.perf_counter()-t
 with torch.no_grad():ts=time.perf_counter();gen=g(torch.randn(1200,24)).numpy();ms=(time.perf_counter()-ts)/1.2
 p,r=pr(gen);rows.append([seed,"GAN",p,r,sec,ms,sum(q.numel() for q in g.parameters())+sum(q.numel() for q in disc.parameters())])
df=pd.DataFrame(rows,columns=["seed","model","support_precision","support_recall","train_seconds","sample_ms_per_image","parameters"]);df.to_csv(OUT/"figure22_runs.csv",index=False);a=df.groupby("model",sort=False).agg(precision_mean=("support_precision","mean"),precision_sd=("support_precision","std"),recall_mean=("support_recall","mean"),recall_sd=("support_recall","std"),train_seconds_mean=("train_seconds","mean"),train_seconds_sd=("train_seconds","std"),sample_ms_mean=("sample_ms_per_image","mean"),sample_ms_sd=("sample_ms_per_image","std"),parameters=("parameters","first")).reset_index();a.to_csv(OUT/"figure22_summary.csv",index=False)
fig,ax=plt.subplots(figsize=(6.8,4.9),dpi=240);colors={"Latent AE":"#66735B","VAE":"#173A52","GAN":"#725B78"};marks={"Latent AE":"s","VAE":"o","GAN":"^"}
for _,q in a.iterrows():ax.errorbar(q.recall_mean,q.precision_mean,xerr=q.recall_sd,yerr=q.precision_sd,fmt=marks[q.model],ms=np.sqrt(42+q.parameters/1200),capsize=3,lw=1,color=colors[q.model],label=f"{q.model} ({int(q.parameters/1000)}k params)")
ax.set(xlabel="Empirical support recall",ylabel="Empirical support precision",xlim=(0,1.02),ylim=(0,1.02));ax.grid(True,lw=.45,alpha=.24);ax.legend(frameon=False,fontsize=8,loc="lower left");fig.tight_layout();fig.savefig(OUT/"figure22.png",bbox_inches="tight");fig.savefig(OUT/"figure22.svg",bbox_inches="tight");plt.close(fig)
