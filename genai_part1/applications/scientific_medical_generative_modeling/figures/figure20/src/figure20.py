from pathlib import Path
import numpy as np,pandas as pd,torch,torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
SEED=20020; np.random.seed(SEED); torch.manual_seed(SEED)
OUT=Path(__file__).resolve().parent.parent/"outputs"; OUT.mkdir(parents=True,exist_ok=True)
raw=load_breast_cancer(); X=raw.data.astype(np.float32); y=raw.target.astype(np.int64); names=np.array(raw.feature_names)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=SEED); scaler=StandardScaler().fit(Xtr); Ztr=scaler.transform(Xtr).astype(np.float32)
loader=DataLoader(TensorDataset(torch.tensor(Ztr)),64,shuffle=True,generator=torch.Generator().manual_seed(SEED))
class VAE(nn.Module):
 def __init__(self):
  super().__init__(); self.enc=nn.Sequential(nn.Linear(30,64),nn.SiLU(),nn.Linear(64,48),nn.SiLU()); self.mu=nn.Linear(48,8); self.lv=nn.Linear(48,8); self.dec=nn.Sequential(nn.Linear(8,48),nn.SiLU(),nn.Linear(48,64),nn.SiLU(),nn.Linear(64,30))
 def forward(self,x):
  h=self.enc(x); m,l=self.mu(h),self.lv(h); z=m+torch.exp(.5*l)*torch.randn_like(m); return self.dec(z),m,l
m=VAE(); opt=torch.optim.Adam(m.parameters(),1.5e-3); log=[]
for _ in range(180):
 total=0
 for (xb,) in loader:
  out,mu,lv=m(xb); loss=nn.functional.mse_loss(out,xb)+.035*(-.5*torch.mean(1+lv-mu.pow(2)-lv.exp())); opt.zero_grad(); loss.backward(); opt.step(); total+=loss.item()
 log.append(total/len(loader))
with torch.no_grad(): gen_std=m.dec(torch.randn(3000,8)).numpy()
gen=scaler.inverse_transform(gen_std); rm=Xtr.mean(0); gm=gen.mean(0); rs=Xtr.std(0,ddof=1); gs=gen.std(0,ddof=1); smd=(gm-rm)/rs; ratio=gs/rs; cr=np.corrcoef(Xtr,rowvar=False); cg=np.corrcoef(gen,rowvar=False)
pd.DataFrame({"feature":names,"standardized_mean_difference":smd,"sd_ratio":ratio}).to_csv(OUT/"figure20_feature_metrics.csv",index=False)
pd.DataFrame({"epoch":range(1,181),"vae_loss":log}).to_csv(OUT/"figure20_training_log.csv",index=False)
pd.DataFrame({"metric":["mean_abs_standardized_mean_difference","median_sd_ratio","correlation_matrix_MAE","generated_samples","training_observations"],"value":[np.mean(abs(smd)),np.median(ratio),np.mean(abs(cr-cg)),len(gen),len(Xtr)]}).to_csv(OUT/"figure20_summary.csv",index=False)
fig=plt.figure(figsize=(7.4,6),dpi=240); grid=fig.add_gridspec(2,2,hspace=.38,wspace=.34); axs=[fig.add_subplot(grid[i,j]) for i in range(2) for j in range(2)]; navy="#173A52";violet="#725B78";olive="#66735B"; idx=np.arange(30)
axs[0].axhline(0,lw=.7,color="0.45");axs[0].plot(idx,smd,"o",ms=3.3,color=navy);axs[0].set(xlabel="Feature index",ylabel="Standardized mean difference")
axs[1].axhline(1,lw=.7,color="0.45");axs[1].plot(idx,ratio,"s",ms=3.1,color=olive);axs[1].set(xlabel="Feature index",ylabel="Generated / real SD")
tri=np.triu_indices_from(cr,k=1);axs[2].scatter(cr[tri],cg[tri],s=8,alpha=.45,color=violet,edgecolors="none");lo=min(cr[tri].min(),cg[tri].min());hi=max(cr[tri].max(),cg[tri].max());axs[2].plot([lo,hi],[lo,hi],lw=.8,color="0.35");axs[2].set(xlabel="Real feature correlation",ylabel="Generated feature correlation")
pca=PCA(n_components=2,random_state=SEED).fit(Ztr);pr=pca.transform(Ztr);pg=pca.transform(gen_std);axs[3].scatter(pr[:,0],pr[:,1],s=8,alpha=.28,color=navy,label="Observed training");axs[3].scatter(pg[:900,0],pg[:900,1],s=8,alpha=.22,color=violet,label="Generated");axs[3].set(xlabel="PC1",ylabel="PC2");axs[3].legend(frameon=False,fontsize=8)
for ax,l in zip(axs,["(a)","(b)","(c)","(d)"]): ax.grid(True,lw=.4,alpha=.22);ax.text(-.15,1.04,l,transform=ax.transAxes,fontweight="bold",fontsize=9)
fig.savefig(OUT/"figure20.png",bbox_inches="tight");fig.savefig(OUT/"figure20.svg",bbox_inches="tight");plt.close(fig)
