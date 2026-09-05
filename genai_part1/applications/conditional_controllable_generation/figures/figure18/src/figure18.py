from pathlib import Path
import numpy as np,pandas as pd,torch,torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
SEED=18018; torch.manual_seed(SEED); np.random.seed(SEED); OUT=Path(__file__).resolve().parent.parent/"outputs"; OUT.mkdir(parents=True,exist_ok=True)
d=load_digits(); X=torch.tensor(d.images/16.,dtype=torch.float32).view(-1,64); y=torch.tensor(d.target,dtype=torch.long)
loader=DataLoader(TensorDataset(X,y),128,shuffle=True,generator=torch.Generator().manual_seed(SEED))
class CVAE(nn.Module):
 def __init__(self):
  super().__init__(); self.zdim=12; self.emb=nn.Embedding(10,10); self.enc=nn.Sequential(nn.Linear(74,128),nn.SiLU()); self.mu=nn.Linear(128,12); self.lv=nn.Linear(128,12); self.dec=nn.Sequential(nn.Linear(22,128),nn.SiLU(),nn.Linear(128,64),nn.Sigmoid())
 def forward(self,x,c):
  e=self.emb(c); h=self.enc(torch.cat([x,e],1)); m,l=self.mu(h),self.lv(h); z=m+torch.exp(.5*l)*torch.randn_like(m); return self.dec(torch.cat([z,e],1)),m,l
 def generate(self,z,c,scale): return self.dec(torch.cat([z,self.emb(c)*scale],1))
m=CVAE(); o=torch.optim.Adam(m.parameters(),2e-3); losses=[]
for _ in range(65):
 t=0
 for xb,cb in loader:
  out,mu,lv=m(xb,cb); loss=nn.functional.binary_cross_entropy(out,xb,reduction="sum")/len(xb)+.12*(-.5*torch.sum(1+lv-mu.pow(2)-lv.exp())/len(xb)); o.zero_grad(); loss.backward(); o.step(); t+=loss.item()
 losses.append(t/len(loader))
clf=nn.Sequential(nn.Linear(64,128),nn.ReLU(),nn.Linear(128,10)); oc=torch.optim.Adam(clf.parameters(),2e-3)
for _ in range(55):
 for xb,cb in loader:
  loss=nn.functional.cross_entropy(clf(xb),cb); oc.zero_grad(); loss.backward(); oc.step()
centroids=torch.stack([X[y==k].mean(0) for k in range(10)]); rows=[]
with torch.no_grad():
 for s in np.arange(0,2.01,.25):
  c=torch.arange(10).repeat_interleave(120); gen=m.generate(torch.randn(len(c),12),c,float(s)); pred=clf(gen).argmax(1); rows.append([s,(pred==c).float().mean().item(),torch.sqrt(((gen-centroids[c])**2).mean(1)).mean().item(),torch.stack([gen[c==k].std(0).mean() for k in range(10)]).mean().item()])
df=pd.DataFrame(rows,columns=["conditioning_scale","class_adherence","target_centroid_rmse","within_class_pixel_std"]); df.to_csv(OUT/"figure18_results.csv",index=False); pd.DataFrame({"epoch":range(1,66),"cvae_loss":losses}).to_csv(OUT/"figure18_training_log.csv",index=False)
fig,ax=plt.subplots(figsize=(7.2,4.7),dpi=240); c1="#173A52";c2="#7A526B";c3="#66735B"; ax.plot(df.conditioning_scale,df.class_adherence,"o-",lw=1.6,color=c1,label="Class adherence"); ax.plot(df.conditioning_scale,df.within_class_pixel_std,"s-",lw=1.4,color=c3,label="Within-class diversity"); ax.set(xlabel="Conditioning embedding scale",ylabel="Adherence / diversity",ylim=(0,1.02)); ax.grid(True,lw=.5,alpha=.25); ax2=ax.twinx(); ax2.plot(df.conditioning_scale,df.target_centroid_rmse,"^-",lw=1.4,color=c2,label="Target-centroid RMSE"); ax2.set_ylabel("Target-centroid RMSE"); h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels(); ax.legend(h1+h2,l1+l2,frameon=False,loc="center right"); fig.tight_layout(); fig.savefig(OUT/"figure18.png",bbox_inches="tight"); fig.savefig(OUT/"figure18.svg",bbox_inches="tight"); plt.close(fig)
