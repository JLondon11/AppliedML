"""Pass 14 closure source for Generative AI Part I figures 4,6,8,9,15,25,26,30,33,34,35."""
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.patches import Rectangle
OUT=Path(__file__).resolve().parent; SEED=1729
def save(fig,n): fig.savefig(OUT/f'Figure_{n:03d}.png',dpi=260,bbox_inches='tight',pad_inches=.03); plt.close(fig)
def mixtures(n,mode):
 rng=np.random.default_rng(SEED+n); real=np.vstack([rng.normal(c,.35,(400,2)) for c in [(-1,-1),(-1,1),(1,-1),(1,1)]])
 if mode=='benchmark':
  fig,a=plt.subplots(figsize=(5.8,3.6)); p=[.72,.81,.84,.88]; r=[.66,.82,.74,.90]; labs=['AE-like','VAE-like','GAN-like','Diffusion-like']; a.scatter(p,r,s=60); [a.text(x+.01,y+.01,l,fontsize=7) for x,y,l in zip(p,r,labs)]; a.set_xlabel('support precision'); a.set_ylabel('support recall')
 elif mode=='samples':
  fig,axs=plt.subplots(1,4,figsize=(8.6,2.2)); ss=[real,rng.normal(0,1,(1200,2)),np.tanh(rng.normal(0,1,(1200,2)))*1.3,real+rng.normal(0,.08,real.shape)]
  for a,s in zip(axs,ss): a.scatter(s[:,0],s[:,1],s=3,alpha=.35); a.set_xticks([]); a.set_yticks([])
 elif mode=='stability':
  e=np.arange(1,81); fig,a=plt.subplots(figsize=(6,3.6));
  for i,l in enumerate(['AE','VAE','GAN']): a.plot(e,.9*np.exp(-e/(12+4*i))+.12+.025*i+.04*rng.normal(size=len(e))*(1 if l!='GAN' else 2),label=l)
  a.legend(frameon=False); a.set_xlabel('epoch'); a.set_ylabel('objective')
 elif mode=='reverse':
  fig,axs=plt.subplots(1,4,figsize=(8.8,2.2));
  for i,a in enumerate(axs): s=rng.normal(0,1,(900,2)); tar=real[rng.choice(len(real),900,replace=False)]; q=(1-[0,.35,.7,1][i])*s+[0,.35,.7,1][i]*tar; a.scatter(q[:,0],q[:,1],s=3,alpha=.35); a.set_xticks([]); a.set_yticks([])
 elif mode=='attention':
  A=rng.random((14,14)); A=A/A.sum(1,keepdims=True); H=rng.normal(size=(14,12)); fig,axs=plt.subplots(1,2,figsize=(7.6,3)); axs[0].imshow(A,cmap='viridis',aspect='auto'); axs[1].imshow(H,cmap='coolwarm',aspect='auto')
 save(fig,n)
mixtures(4,'benchmark'); mixtures(6,'samples'); mixtures(8,'samples'); mixtures(9,'stability')
d=load_breast_cancer(); X=StandardScaler().fit_transform(d.data); Z=PCA(2).fit_transform(X); fig,axs=plt.subplots(1,2,figsize=(7.5,3)); axs[0].scatter(Z[:,0],Z[:,1],c=d.target,cmap='coolwarm',s=9); axs[1].hist(np.linalg.norm(X,axis=1),bins=25); save(fig,15)
steps=np.array([10,20,50,100,200]); fig,a=plt.subplots(figsize=(6,3.6));
for name,y in {'linear':.34/np.sqrt(steps)+.05,'cosine':.29/np.sqrt(steps)+.045,'variance-preserving':.31/np.sqrt(steps)+.048}.items(): a.plot(steps,y,marker='o',label=name)
a.legend(frameon=False); a.set_xlabel('sampling steps'); a.set_ylabel('distribution error'); save(fig,25)
mixtures(26,'reverse'); mixtures(30,'attention')
rng=np.random.default_rng(SEED+33); fig,axs=plt.subplots(1,2,figsize=(7.4,3)); pts=np.vstack([rng.normal([-2,1],[.8,.35],(180,2)),rng.normal([1.5,-1],[.55,.25],(120,2)),rng.uniform([-4,-3],[4,3],(300,2))]); axs[0].scatter(pts[:,0],pts[:,1],s=3); axs[1].scatter(pts[:,0],pts[:,1],s=3); axs[1].add_patch(Rectangle((-3,.25),2,1.5,fill=False)); save(fig,33)
fig,a=plt.subplots(figsize=(5.5,3.7)); fps=[78,52,28,11]; m=[48,57,64,68]; a.scatter(fps,m,s=60); a.set_xlabel('throughput (frames/s)'); a.set_ylabel('mAP-like score (%)'); save(fig,34)
H,W=100,150; yy,xx=np.mgrid[:H,:W]; m=np.zeros((H,W),int); m[((xx-45)**2/28**2+(yy-55)**2/20**2)<1]=1; m[((xx-105)**2/23**2+(yy-48)**2/26**2)<1]=2; noisy=m.copy(); flip=np.random.default_rng(SEED).random((H,W))<.1; noisy[flip]=np.random.default_rng(SEED+1).integers(0,3,flip.sum()); fig,axs=plt.subplots(1,3,figsize=(8.5,2.6)); axs[0].imshow(.2+.2*np.sin(xx/15),cmap='gray'); axs[1].imshow(noisy,cmap='viridis'); axs[2].imshow(m,cmap='viridis'); [a.axis('off') for a in axs]; save(fig,35)
