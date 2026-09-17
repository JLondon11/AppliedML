"""Exact Pass 12 generators for Generative AI Part I figures 3, 10, 11, 14, 20, 24.

Sections:
- Fig 3: Latent Variables and Distributed Representations
- Fig 10: Probabilistic Encoder--Decoder Structure
- Fig 11: Evidence Lower Bound and Training Objective
- Fig 14: Hierarchical Latent Structure and Practical Evaluation
- Fig 20: Stability and Latent Control
- Fig 24: DDPM, DDIM, and Score-Based Models

These are the code paths used for the accepted Pass 12 remediations.
"""
from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
np.random.seed(13); random.seed(13); torch.manual_seed(13)

def save(fig, n):
    fig.savefig(OUT / f"figure_{n:02d}.png", dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

def support_pr(real, gen, k=5):
    nr=NearestNeighbors(n_neighbors=k+1).fit(real)
    dr,_=nr.kneighbors(real); rr=dr[:,-1]
    q=NearestNeighbors(n_neighbors=1).fit(real)
    d,idx=q.kneighbors(gen)
    precision=np.mean(d[:,0] <= rr[idx[:,0]])
    ng=NearestNeighbors(n_neighbors=k+1).fit(gen)
    dg,_=ng.kneighbors(gen); rg=dg[:,-1]
    q2=NearestNeighbors(n_neighbors=1).fit(gen)
    d2,idx2=q2.kneighbors(real)
    recall=np.mean(d2[:,0] <= rg[idx2[:,0]])
    return precision, recall

def figure03():
    rng=np.random.default_rng(13)
    centers=np.array([[-2,-2],[-2,2],[2,-2],[2,2]])
    real=np.vstack([rng.normal(c,.38,(600,2)) for c in centers])
    gens={
        'mode drop':np.vstack([rng.normal(c,.38,(800,2)) for c in centers[:3]]),
        'blurred':np.vstack([rng.normal(c,.72,(600,2)) for c in centers]),
        'shifted':np.vstack([rng.normal(c+[.45,.15],.38,(600,2)) for c in centers]),
        'matched':np.vstack([rng.normal(c,.38,(600,2)) for c in centers]),
    }
    fig,axs=plt.subplots(1,2,figsize=(7.6,3.0))
    for name,g in gens.items():
        p,r=support_pr(real,g); axs[0].scatter(p,r,s=55,label=name)
    axs[0].set(xlim=(0,1.02),ylim=(0,1.02),xlabel='support precision',ylabel='support recall')
    axs[0].legend(frameon=False,fontsize=7)
    axs[1].scatter(real[:,0],real[:,1],s=3,alpha=.2,label='real')
    axs[1].scatter(gens['mode drop'][:,0],gens['mode drop'][:,1],s=3,alpha=.2,label='mode drop')
    axs[1].set_aspect('equal'); axs[1].legend(frameon=False,fontsize=7)
    save(fig,3)

class VAE(nn.Module):
    def __init__(self):
        super().__init__(); self.e1=nn.Linear(64,48); self.mu=nn.Linear(48,2); self.lv=nn.Linear(48,2); self.d1=nn.Linear(2,48); self.out=nn.Linear(48,64)
    def encode(self,x):
        h=torch.tanh(self.e1(x)); return self.mu(h),self.lv(h)
    def decode(self,z): return torch.sigmoid(self.out(torch.tanh(self.d1(z))))
    def forward(self,x):
        mu,lv=self.encode(x); z=mu+torch.randn_like(mu)*torch.exp(.5*lv); return self.decode(z),mu,lv

def train_vae():
    d=load_digits(); X=d.data.astype('float32')/16.; y=d.target.astype(int)
    Xtr,Xva,ytr,yva=train_test_split(X,y,test_size=.25,random_state=13,stratify=y)
    tr=DataLoader(TensorDataset(torch.tensor(Xtr)),batch_size=128,shuffle=True)
    va=DataLoader(TensorDataset(torch.tensor(Xva)),batch_size=256,shuffle=False)
    m=VAE(); opt=torch.optim.Adam(m.parameters(),lr=2e-3)
    th=[]; vh=[]; rh=[]; kh=[]
    for _ in range(60):
        m.train(); tot=rec=kl=n=0
        for (xb,) in tr:
            opt.zero_grad(); xr,mu,lv=m(xb); r=F.binary_cross_entropy(xr,xb,reduction='sum'); k=-.5*torch.sum(1+lv-mu.pow(2)-lv.exp()); (r+k).backward(); opt.step(); tot+=(r+k).item(); rec+=r.item(); kl+=k.item(); n+=len(xb)
        th.append(tot/n); rh.append(rec/n); kh.append(kl/n)
        m.eval(); vt=vn=0
        with torch.no_grad():
            for (xb,) in va:
                xr,mu,lv=m(xb); loss=F.binary_cross_entropy(xr,xb,reduction='sum')-.5*torch.sum(1+lv-mu.pow(2)-lv.exp()); vt+=loss.item(); vn+=len(xb)
        vh.append(vt/vn)
    return m,Xva,yva,th,vh,rh,kh

def figures10_11_14():
    m,Xva,yva,th,vh,rh,kh=train_vae(); m.eval()
    with torch.no_grad():
        i1=np.where(yva==1)[0][0]; i7=np.where(yva==7)[0][0]
        za=m.encode(torch.tensor(Xva[i1:i1+1]))[0]; zb=m.encode(torch.tensor(Xva[i7:i7+1]))[0]
        a=torch.linspace(0,1,8)[:,None]; imgs=m.decode((1-a)*za+a*zb).reshape(-1,8,8).numpy()
    fig,axs=plt.subplots(1,8,figsize=(8.8,1.45))
    for ax,img in zip(axs,imgs): ax.imshow(img,cmap='gray',vmin=0,vmax=1); ax.axis('off')
    save(fig,10)
    fig,axs=plt.subplots(1,2,figsize=(7.8,3.0)); axs[0].plot(th,label='train negative ELBO'); axs[0].plot(vh,label='validation negative ELBO'); axs[0].legend(frameon=False,fontsize=7); axs[0].set_xlabel('epoch'); axs[0].set_ylabel('loss per sample'); axs[1].plot(rh,label='reconstruction'); axs[1].plot(kh,label='KL'); axs[1].legend(frameon=False,fontsize=7); axs[1].set_xlabel('epoch'); axs[1].set_ylabel('term per sample'); save(fig,11)
    with torch.no_grad(): Z=m.encode(torch.tensor(Xva))[0].numpy()
    idx=NearestNeighbors(n_neighbors=11).fit(Z).kneighbors(Z,return_distance=False)[:,1:]; purity=np.mean(yva[idx]==yva[:,None],axis=1)
    fig,axs=plt.subplots(1,2,figsize=(7.8,3.0)); axs[0].scatter(Z[:,0],Z[:,1],c=yva,cmap='tab10',s=9,alpha=.8,linewidths=0); axs[0].set(xlabel='latent z1',ylabel='latent z2'); axs[1].hist(purity,bins=np.linspace(0,1,11)); axs[1].axvline(purity.mean(),ls='--',lw=1); axs[1].set(xlabel='10-NN label purity',ylabel='validation samples'); save(fig,14)

def figure20():
    steps=np.arange(100); fig,axs=plt.subplots(1,3,figsize=(9,2.8)); axs[0].plot(steps,1.1+.35*np.sin(steps/4)*np.exp(-steps/80),label='saturating adversarial proxy'); axs[0].plot(steps,.8*np.exp(-steps/35)+.22,label='Wasserstein proxy'); axs[0].legend(frameon=False,fontsize=6); z=np.linspace(-2,2,80); axs[1].plot(z,np.tanh(z)); axs[1].plot(z,1.4*np.tanh(z)+.2); x=np.linspace(0,1,128); c=np.sin(2*np.pi*x); m=.35*np.sin(8*np.pi*x); f=.12*np.sin(32*np.pi*x); axs[2].plot(x,c); axs[2].plot(x,c+m); axs[2].plot(x,c+m+f); save(fig,20)

def figure24():
    rng=np.random.default_rng(24); x=np.linspace(0,1,128); clean=np.sin(2*np.pi*x)+.45*np.sin(6*np.pi*x); sigmas=[0,.25,.55,.9]; fig,axs=plt.subplots(1,4,figsize=(9,2.5))
    for i,(ax,s) in enumerate(zip(axs,sigmas)):
        noisy=np.sqrt(max(0,1-s*s))*clean+s*rng.normal(size=len(x)); ax.plot(x,noisy,lw=1); ax.set_xticks([]); ax.set_yticks([])
        if i==3: ax.plot(x,clean,lw=1,alpha=.6)
    save(fig,24)

if __name__=='__main__':
    figure03(); figures10_11_14(); figure20(); figure24()
