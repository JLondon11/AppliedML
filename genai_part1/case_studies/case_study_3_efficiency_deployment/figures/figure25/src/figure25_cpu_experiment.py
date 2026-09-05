from pathlib import Path
import os,time,json,platform,sys,numpy as np,pandas as pd,torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

SEED=25025; SEEDS=[25025,25026,25027]; STEPS=[10,20,40,80]; SAMPLERS=["DDPM","DDIM"]
NGEN=1200; T=80; torch.manual_seed(SEED); np.random.seed(SEED)
HERE=Path(__file__).resolve().parent; OUT=HERE.parent/"outputs"; OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(max(1,min(4,os.cpu_count() or 1)))

d=load_digits(); X=(d.images.astype(np.float32)/16).reshape(-1,64); y=d.target.astype(np.int64)
tr,te=train_test_split(np.arange(len(X)),test_size=.25,stratify=y,random_state=SEED)
Xtr=torch.tensor(X[tr]); Xte=torch.tensor(X[te]); ytr=torch.tensor(y[tr]); yte=torch.tensor(y[te])
loader=DataLoader(TensorDataset(Xtr),128,shuffle=True,generator=torch.Generator().manual_seed(SEED))
cloader=DataLoader(TensorDataset(Xtr,ytr),128,shuffle=True,generator=torch.Generator().manual_seed(SEED+1))

class C(nn.Module):
 def __init__(self): super().__init__(); self.f=nn.Sequential(nn.Linear(64,128),nn.ReLU(),nn.Linear(128,48),nn.ReLU()); self.h=nn.Linear(48,10)
 def forward(self,x,feat=False): z=self.f(x); return z if feat else self.h(z)
clf=C(); o=torch.optim.Adam(clf.parameters(),2e-3)
for _ in range(55):
 for xb,yb in cloader:
  loss=nn.functional.cross_entropy(clf(xb),yb); o.zero_grad(); loss.backward(); o.step()
with torch.no_grad(): acc=(clf(Xte).argmax(1)==yte).float().mean().item(); rf=clf(Xte,True).numpy()

betas=torch.linspace(1e-4,.08,T); a=1-betas; abar=torch.cumprod(a,0)
class E(nn.Module):
 def __init__(self): super().__init__(); self.n=nn.Sequential(nn.Linear(65,192),nn.SiLU(),nn.Linear(192,192),nn.SiLU(),nn.Linear(192,64))
 def forward(self,x,t): return self.n(torch.cat([x,(t.float()/(T-1)).unsqueeze(1)],1))
eps=E(); o=torch.optim.Adam(eps.parameters(),1e-3); log=[]
for ep in range(80):
 tot=0
 for (x0,) in loader:
  bs=len(x0); ti=torch.randint(0,T,(bs,)); n=torch.randn_like(x0); ab=abar[ti].unsqueeze(1); xt=ab.sqrt()*x0+(1-ab).sqrt()*n
  loss=nn.functional.mse_loss(eps(xt,ti),n); o.zero_grad(); loss.backward(); o.step(); tot+=loss.item()
 log.append([ep+1,tot/len(loader)])
pd.DataFrame(log,columns=["epoch","noise_mse"]).to_csv(OUT/"figure25_training_log.csv",index=False)

def fd(r,f):
 m1,m2=r.mean(0),f.mean(0); c1,c2=np.cov(r,rowvar=False),np.cov(f,rowvar=False); cm=sqrtm(c1@c2); cm=cm.real if np.iscomplexobj(cm) else cm
 return float(np.sum((m1-m2)**2)+np.trace(c1+c2-2*cm))
def sched(s):
 out=[]
 for t in np.linspace(T-1,0,s,dtype=int):
  if int(t) not in out: out.append(int(t))
 return out
@torch.no_grad()
def ddim(n,s,seed):
 g=torch.Generator().manual_seed(seed); x=torch.randn(n,64,generator=g); ts=sched(s)
 for i,t in enumerate(ts):
  tv=torch.full((n,),t,dtype=torch.long); e=eps(x,tv); at=abar[t]; x0=((x-torch.sqrt(1-at)*e)/torch.sqrt(at)).clamp(0,1)
  x=x0 if i==len(ts)-1 else torch.sqrt(abar[ts[i+1]])*x0+torch.sqrt(1-abar[ts[i+1]])*e
 return x.clamp(0,1),len(ts)
@torch.no_grad()
def ddpm(n,s,seed):
 g=torch.Generator().manual_seed(seed); x=torch.randn(n,64,generator=g); ts=sched(s)
 for i,t in enumerate(ts):
  tv=torch.full((n,),t,dtype=torch.long); e=eps(x,tv); at=abar[t]; x0=((x-torch.sqrt(1-at)*e)/torch.sqrt(at)).clamp(0,1)
  if i==len(ts)-1: x=x0
  else:
   j=ts[i+1]; aj=abar[j]; sig2=torch.clamp(((1-aj)/(1-at))*(1-at/aj),min=0); x=torch.sqrt(aj)*x0+torch.sqrt(torch.clamp(1-aj-sig2,min=0))*e+torch.sqrt(sig2)*torch.randn(x.shape,generator=g)
 return x.clamp(0,1),len(ts)

rows=[]
for name,fn in [("DDPM",ddpm),("DDIM",ddim)]:
 for s in STEPS:
  for rep,seed in enumerate(SEEDS,1):
   t0=time.perf_counter(); fake,nfe=fn(NGEN,s,seed+s*100); sec=time.perf_counter()-t0
   with torch.no_grad(): ff=clf(fake,True).numpy()
   rows.append([name,s,nfe,rep,seed,NGEN,sec,1000*sec/NGEN,NGEN/sec,fd(rf,ff)])
runs=pd.DataFrame(rows,columns=["sampler","requested_steps","actual_nfe_per_sample","repeat","seed","generated_samples","sampling_seconds","latency_ms_per_image","throughput_images_per_second","feature_frechet_distance"])
runs.to_csv(OUT/"figure25_runs.csv",index=False)
q=runs.groupby(["sampler","requested_steps"],as_index=False).agg(ffd_mean=("feature_frechet_distance","mean"),ffd_sd=("feature_frechet_distance","std"),latency_ms_mean=("latency_ms_per_image","mean"),latency_ms_sd=("latency_ms_per_image","std"),throughput_mean=("throughput_images_per_second","mean"),nfe_mean=("actual_nfe_per_sample","mean"))
def dom(lat,qual):
 z=np.zeros(len(lat),bool)
 for i in range(len(lat)):
  for j in range(len(lat)):
   if i!=j and lat[j]<=lat[i] and qual[j]<=qual[i] and (lat[j]<lat[i] or qual[j]<qual[i]): z[i]=True; break
 return z
q["dominated"]=dom(q.latency_ms_mean.to_numpy(),q.ffd_mean.to_numpy()); q.to_csv(OUT/"figure25_summary.csv",index=False)
(OUT/"figure25_environment.json").write_text(json.dumps({"platform":platform.platform(),"python":sys.version,"torch":torch.__version__,"numpy":np.__version__,"device":"CPU","cpu_count_visible":os.cpu_count(),"torch_num_threads":torch.get_num_threads(),"dataset":"sklearn digits","train_observations":len(tr),"test_observations":len(te),"classifier_test_accuracy":acc,"generated_samples_per_run":NGEN,"repeats":len(SEEDS),"samplers":SAMPLERS,"step_budgets":STEPS,"quality_metric":"Frechet distance in learned 48-D classifier feature space"},indent=2),encoding="utf-8")

colors={"DDPM":"#173A52","DDIM":"#725B78"}; marks={"DDPM":"o","DDIM":"s"}; fig,axs=plt.subplots(1,2,figsize=(7.6,3.85),dpi=240)
ax=axs[0]; D=q[q.dominated]
if len(D): ax.errorbar(D.latency_ms_mean,D.ffd_mean,xerr=D.latency_ms_sd,yerr=D.ffd_sd,fmt="o",ms=4,color=".7",ecolor=".78",capsize=2,lw=.7,label="Pareto-dominated")
for name,g in q.groupby("sampler"):
 ax.errorbar(g.latency_ms_mean,g.ffd_mean,xerr=g.latency_ms_sd,yerr=g.ffd_sd,fmt=marks[name],ms=5.2,capsize=2.2,lw=.9,color=colors[name],label=name)
 for _,r in g.iterrows(): ax.annotate(str(int(r.requested_steps)),(r.latency_ms_mean,r.ffd_mean),xytext=(4,4),textcoords="offset points",fontsize=6.5,color=colors[name])
ax.set(xlabel="Measured CPU sampling latency (ms/image)",ylabel="Feature Frechet distance (lower is better)"); ax.grid(True,lw=.4,alpha=.22); ax.legend(frameon=False,fontsize=7)
ax=axs[1]
for name,g in q.groupby("sampler"):
 g=g.sort_values("nfe_mean"); ax.errorbar(g.nfe_mean,g.ffd_mean,yerr=g.ffd_sd,fmt=marks[name]+"-",ms=5,capsize=2.2,lw=.9,color=colors[name])
ax.set(xlabel="Measured denoiser evaluations per sample",ylabel="Feature Frechet distance (lower is better)"); ax.grid(True,lw=.4,alpha=.22)
for ax in axs:
 for sp in ax.spines.values(): sp.set_linewidth(.55)
fig.tight_layout(); fig.savefig(OUT/"figure25.svg",bbox_inches="tight"); fig.savefig(OUT/"figure25.pdf",bbox_inches="tight"); fig.savefig(OUT/"figure25.png",dpi=300,bbox_inches="tight")
