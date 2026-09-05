from pathlib import Path
import numpy as np,pandas as pd,torch,torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
BASE=24024;SEEDS=[24024,24025,24026,24027,24028];EPOCHS=85;OUT=Path(__file__).resolve().parent.parent/"outputs";OUT.mkdir(parents=True,exist_ok=True)
d=load_digits();X=(d.images.astype(np.float32)/16).reshape(-1,64);y=d.target.astype(np.int64);tr,te=train_test_split(np.arange(len(X)),test_size=.2,stratify=y,random_state=BASE);Xtr=torch.tensor(X[tr]);Xte=torch.tensor(X[te]);ytr=y[tr];yte=y[te]
class VAE(nn.Module):
 def __init__(self):super().__init__();self.h=nn.Sequential(nn.Linear(64,96),nn.SiLU());self.mu=nn.Linear(96,12);self.lv=nn.Linear(96,12);self.dec=nn.Sequential(nn.Linear(12,96),nn.SiLU(),nn.Linear(96,64),nn.Sigmoid())
 def encode(self,x):h=self.h(x);return self.mu(h),self.lv(h)
 def forward(self,x):m,l=self.encode(x);return self.dec(m+torch.exp(.5*l)*torch.randn_like(m)),m,l
rows=[];rep=None
for seed in SEEDS:
 torch.manual_seed(seed);np.random.seed(seed);loader=DataLoader(TensorDataset(Xtr),128,shuffle=True,generator=torch.Generator().manual_seed(seed));m=VAE();o=torch.optim.Adam(m.parameters(),2e-3)
 for _ in range(EPOCHS):
  for (xb,) in loader:
   out,mu,lv=m(xb);loss=nn.functional.binary_cross_entropy(out,xb,reduction="sum")/len(xb)+.12*(-.5*torch.sum(1+lv-mu.pow(2)-lv.exp())/len(xb));o.zero_grad();loss.backward();o.step()
 with torch.no_grad():ztr=m.encode(Xtr)[0].numpy();zte=m.encode(Xte)[0].numpy()
 ind=NearestNeighbors(n_neighbors=15).fit(ztr).kneighbors(zte,return_distance=False);pur=(ytr[ind]==yte[:,None]).mean(1);sil=silhouette_score(zte,yte);cent=np.stack([ztr[ytr==k].mean(0) for k in range(10)]);between=np.mean([np.linalg.norm(cent[i]-cent[j]) for i in range(10) for j in range(i+1,10)]);within=np.mean([np.linalg.norm(ztr[ytr==k]-cent[k],axis=1).mean() for k in range(10)]);rows.append([seed,pur.mean(),pur.std(ddof=1),sil,between,within,between/within])
 if seed==BASE:rep=(zte,pur)
df=pd.DataFrame(rows,columns=["seed","knn15_purity_mean","knn15_purity_sd","silhouette","between_centroid_distance","within_class_dispersion","separation_ratio"]);df.to_csv(OUT/"figure24_runs.csv",index=False);s=pd.DataFrame({"metric":["knn15_purity","silhouette","separation_ratio"],"mean":[df.knn15_purity_mean.mean(),df.silhouette.mean(),df.separation_ratio.mean()],"sd":[df.knn15_purity_mean.std(ddof=1),df.silhouette.std(ddof=1),df.separation_ratio.std(ddof=1)]});s.to_csv(OUT/"figure24_summary.csv",index=False)
zte,pur=rep;proj=PCA(n_components=2,random_state=BASE).fit_transform(zte);fig,axs=plt.subplots(1,2,figsize=(7.4,3.65),dpi=240);cm=plt.get_cmap("tab10")
for k in range(10):
 q=yte==k;axs[0].scatter(proj[q,0],proj[q,1],s=11,alpha=.62,color=cm(k),edgecolors="none",label=str(k))
axs[0].set(xlabel="PCA score 1",ylabel="PCA score 2");axs[0].legend(frameon=False,ncol=2,fontsize=7,title="Digit",title_fontsize=7)
means=[];sds=[]
for k in range(10):v=pur[yte==k];means.append(v.mean());sds.append(v.std(ddof=1))
axs[1].errorbar(range(10),means,yerr=sds,fmt="o",ms=4.5,capsize=2.5,lw=1,color="#173A52");axs[1].axhline(np.mean(pur),lw=.9,ls="--",color="#725B78",label="Overall mean");axs[1].set(xlabel="Held-out digit class",ylabel="15-NN latent neighborhood purity",ylim=(0,1.03));axs[1].set_xticks(range(10));axs[1].legend(frameon=False,fontsize=8)
for ax in axs:ax.grid(True,lw=.4,alpha=.2)
fig.tight_layout();fig.savefig(OUT/"figure24.png",bbox_inches="tight");fig.savefig(OUT/"figure24.svg",bbox_inches="tight");plt.close(fig)
