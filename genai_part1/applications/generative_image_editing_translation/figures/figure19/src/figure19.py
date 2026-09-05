from pathlib import Path
import numpy as np,pandas as pd,torch,torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.datasets import load_digits
from scipy.ndimage import sobel
import matplotlib.pyplot as plt
SEED=19019; torch.manual_seed(SEED); np.random.seed(SEED); OUT=Path(__file__).resolve().parent.parent/"outputs"; OUT.mkdir(parents=True,exist_ok=True)
d=load_digits(); X=torch.tensor(d.images/16.,dtype=torch.float32).view(-1,64); y=torch.tensor(d.target,dtype=torch.long); loader=DataLoader(TensorDataset(X,y),128,shuffle=True,generator=torch.Generator().manual_seed(SEED))
class AE(nn.Module):
 def __init__(self): super().__init__(); self.enc=nn.Sequential(nn.Linear(64,96),nn.SiLU(),nn.Linear(96,20)); self.dec=nn.Sequential(nn.Linear(20,96),nn.SiLU(),nn.Linear(96,64),nn.Sigmoid())
 def forward(self,x): return self.dec(self.enc(x))
ae=AE(); opt=torch.optim.Adam(ae.parameters(),2e-3); train=[]
for _ in range(70):
 t=0
 for xb,_ in loader:
  pred=ae(xb); loss=nn.functional.mse_loss(pred,xb); opt.zero_grad(); loss.backward(); opt.step(); t+=loss.item()
 train.append(t/len(loader))
sel=[int(torch.where(y==digit)[0][4]) for digit in [2,3,5,8]]; src=X[sel].clone(); labels=y[sel].numpy(); mask=np.zeros((8,8),np.float32); mask[2:6,2:6]=1; mt=torch.tensor(mask).view(1,64)
targets=[]
for k in range(4):
 t=np.zeros((8,8),np.float32)
 if k==0:t[3:5,2:6]=.95
 elif k==1:t[2:6,3:5]=.95
 elif k==2:
  for q in range(2,6):t[q,q]=.95
 else:t[2:6,2]=.9;t[2:6,5]=.9;t[2,2:6]=.9;t[5,2:6]=.9
 targets.append(t)
target=torch.tensor(np.stack(targets)).view(-1,64)
with torch.no_grad(): z0=ae.enc(src)
z=z0.clone().detach().requires_grad_(True); oz=torch.optim.Adam([z],lr=.055)
for _ in range(450):
 out=ae.dec(z); loss=4*(((out-src)*(1-mt))**2).mean()+2.5*(((out-target)*mt)**2).mean()+.002*((z-z0)**2).mean(); oz.zero_grad(); loss.backward(); oz.step()
with torch.no_grad(): edited=ae.dec(z).clamp(0,1)
rows=[]
for i in range(4):
 s=src[i].view(8,8).numpy(); e=edited[i].view(8,8).numpy(); t=targets[i]; outside=mask==0; inside=mask==1; se=np.hypot(sobel(s,0),sobel(s,1)); ee=np.hypot(sobel(e,0),sobel(e,1)); rows.append([int(labels[i]),np.sqrt(np.mean((e[outside]-s[outside])**2)),np.sqrt(np.mean((e[inside]-t[inside])**2)),np.sqrt(np.mean((ee[outside]-se[outside])**2))])
pd.DataFrame(rows,columns=["source_digit","outside_mask_rmse","control_region_rmse","outside_edge_rmse"]).to_csv(OUT/"figure19_results.csv",index=False); pd.DataFrame({"epoch":range(1,71),"autoencoder_mse":train}).to_csv(OUT/"figure19_training_log.csv",index=False); np.savez_compressed(OUT/"figure19_samples.npz",source=src.numpy(),control=np.stack(targets),edited=edited.numpy(),mask=mask)
fig,axs=plt.subplots(4,4,figsize=(6.6,6),dpi=240); cols=["Source","Control","Edited","|Residual|"]
for i in range(4):
 s=src[i].view(8,8).numpy(); e=edited[i].view(8,8).numpy()
 for j,img in enumerate([s,targets[i],e,np.abs(e-s)]):
  axs[i,j].imshow(img,cmap="gray",vmin=0,vmax=1,interpolation="nearest"); axs[i,j].set_xticks([]); axs[i,j].set_yticks([])
  if i==0: axs[i,j].set_title(cols[j],fontsize=9,pad=4)
 axs[i,0].set_ylabel(f"digit {labels[i]}",rotation=0,ha="right",va="center",labelpad=12,fontsize=9)
plt.subplots_adjust(wspace=.08,hspace=.08,left=.11,right=.995,top=.95,bottom=.02); fig.savefig(OUT/"figure19.png",bbox_inches="tight"); fig.savefig(OUT/"figure19.svg",bbox_inches="tight"); plt.close(fig)
