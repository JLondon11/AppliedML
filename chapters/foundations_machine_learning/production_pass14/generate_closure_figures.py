"""Pass 14 closure source for Foundations of Machine Learning figures 10,25,28,29,31."""
from pathlib import Path
import time, numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
OUT=Path(__file__).resolve().parent; SEED=1729
def save(fig,n): fig.savefig(OUT/f'Figure_{n:03d}.png',dpi=260,bbox_inches='tight',pad_inches=.03); plt.close(fig)
d=load_digits(); X=d.data[:700]; y=d.target[:700]; embs=[PCA(2,random_state=SEED).fit_transform(X),TSNE(2,random_state=SEED,init='pca',learning_rate='auto',perplexity=30,max_iter=750).fit_transform(X),Isomap(n_components=2,n_neighbors=12).fit_transform(X)]; fig,axs=plt.subplots(1,3,figsize=(9,2.8));
for a,E in zip(axs,embs): a.scatter(E[:,0],E[:,1],c=y,cmap='tab10',s=6,linewidths=0); a.set_xticks([]); a.set_yticks([])
save(fig,10)
rng=np.random.default_rng(SEED); n=1200; yy=rng.integers(0,2,n); F=np.c_[rng.normal(1.2*yy,.8,n),rng.normal(.8*yy,.9,n),rng.normal(.5*yy,1,n)]; Xt,Xs,yt,ys=train_test_split(F,yy,test_size=.3,random_state=SEED,stratify=yy); p=LogisticRegression().fit(Xt,yt).predict_proba(Xs)[:,1]; fpr,tpr,_=roc_curve(ys,p); fig,axs=plt.subplots(1,2,figsize=(7.6,3)); axs[0].scatter(Xs[:,0],Xs[:,1],c=ys,cmap='coolwarm',s=10); axs[1].plot(fpr,tpr); axs[1].plot([0,1],[0,1],'--'); save(fig,25)
x=np.linspace(0,2*np.pi,160); t=np.linspace(0,2.4,110); Xg,T=np.meshgrid(x,t); ref=np.sin(Xg-1.1*T)+.35*np.sin(3*(Xg-.7*T)); pred=np.sin(Xg-1.05*T)+.33*np.sin(3*(Xg-.68*T)); fig,axs=plt.subplots(1,3,figsize=(9,2.8));
for a,A in zip(axs,[ref,pred,np.abs(ref-pred)]): a.imshow(A,aspect='auto',origin='lower',cmap='viridis')
save(fig,28)
sizes=np.array([64,128,256,512]); lat=[]
for n in sizes:
 a=rng.normal(size=(n,n)).astype('float32'); b=rng.normal(size=(n,n)).astype('float32'); ts=[]
 for _ in range(3): s=time.perf_counter(); _=a@b; ts.append((time.perf_counter()-s)*1000)
 lat.append(np.median(ts))
fig,axs=plt.subplots(1,2,figsize=(7.6,3)); axs[0].plot(sizes,lat,marker='o'); axs[1].plot(sizes,1/np.maximum(lat,1e-6),marker='s'); save(fig,29)
fig,axs=plt.subplots(1,2,figsize=(7.7,3));
for u in range(6): c=np.arange(1,180); deg=np.maximum(0,(c-(65+u*7))/(120-u*5)); axs[0].plot(c,1-.55*deg+.04*rng.normal(size=len(c)),lw=.9)
true=np.linspace(0,110,160); axs[1].scatter(true,true+rng.normal(0,9,len(true)),s=9); axs[1].plot([0,110],[0,110],'--'); save(fig,31)
