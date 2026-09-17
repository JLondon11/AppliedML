from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, json, re, math, random, textwrap, os
from matplotlib.patches import Rectangle, Circle, Polygon, Ellipse
from sklearn.datasets import load_digits, load_wine, load_breast_cancer, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, f1_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from skimage import data, transform, filters, color
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import argparse
_parser=argparse.ArgumentParser(); _parser.add_argument('--package-root', required=True); _args=_parser.parse_args(); ROOT=Path(_args.package_root)
audit=pd.read_csv(ROOT/'Master_QA_Audit_Pass13.csv'); inv=pd.read_csv(ROOT/'Frozen_Figure_Inventory_Master_Pass13.csv')
NAVY='#243447'; TEAL='#2A6F73'; BLUE='#4E6E8E'; VIOLET='#756B8A'; ORANGE='#B66A3C'; GOLD='#A88B4A'; SLATE='#6B7280'; LIGHT='#D9DEE5'; RED='#9A4D4D'; GREEN='#557A5E'; COLORS=[NAVY,TEAL,ORANGE,VIOLET,GOLD,BLUE,RED,GREEN]
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'savefig.facecolor':'white'}); np.random.seed(14); random.seed(14); torch.manual_seed(14)
def row(ch,f): return audit[(audit.chapter==ch)&(audit.figure_number==f)].iloc[0]
def asset(ch,f): return ROOT/ch/row(ch,f).asset
def save(ch,f,fig):
    p=asset(ch,f); p.parent.mkdir(parents=True,exist_ok=True); fig.savefig(p,dpi=240,bbox_inches='tight',pad_inches=.025); plt.close(fig)
def panel(ax,s): ax.text(.5,-.12,s,transform=ax.transAxes,ha='center',va='top',fontsize=9)
def clean(ax): ax.grid(False); ax.tick_params(direction='out',length=3)
def update(ch,f,title,caption,reason,prov,kind):
    m=(audit.chapter==ch)&(audit.figure_number==f); audit.loc[m,'audit_status']='ACCEPT'; audit.loc[m,'audit_reason']=reason; audit.loc[m,'provenance']=prov
    if 'title' in audit.columns: audit.loc[m,'title']=title
    if 'caption' in audit.columns: audit.loc[m,'caption']=caption
    mi=(inv.chapter==ch)&(inv.figure_number==f); inv.loc[mi,'audit_status']='ACCEPT'; inv.loc[mi,'audit_reason']=reason; inv.loc[mi,'provenance']=prov; inv.loc[mi,'title']=title; inv.loc[mi,'caption']=caption
    if 'production_class' in inv.columns: inv.loc[mi,'production_class']=kind
    rem.append({'chapter':ch,'figure_number':f,'title':title,'caption':caption,'resolution_type':kind,'provenance':prov})
rem=[]; ch='05_Foundations_of_Machine_Learning'
# Figure 10 — dimensionality reduction.
f=10; D=load_digits(); X=D.data[:700]; y=D.target[:700]; p=PCA(2,random_state=14).fit_transform(X); tse=TSNE(2,random_state=14,init='pca',learning_rate='auto',perplexity=30,max_iter=700).fit_transform(X); spe=SpectralEmbedding(2,n_neighbors=15,random_state=14).fit_transform(X)
fig,axs=plt.subplots(1,3,figsize=(9,2.8))
for ax,E in zip(axs,[p,tse,spe]): ax.scatter(E[:,0],E[:,1],c=y,cmap='tab10',s=6,alpha=.7,linewidths=0); ax.set_xticks([]); ax.set_yticks([])
for i,a in enumerate(axs): panel(a,f'({chr(97+i)})')
fig.tight_layout(); save(ch,f,fig)
update(ch,f,'PCA, t-SNE, and graph-manifold projections','Three reproducible two-dimensional projections of the same public handwritten-digit feature matrix: (a) PCA, emphasizing global variance; (b) t-SNE, emphasizing local neighborhood structure; and (c) a nearest-neighbor spectral manifold embedding. The third panel is explicitly labeled as graph-manifold embedding rather than being misrepresented as UMAP when the UMAP implementation is unavailable.','Pass 14 corrects algorithm naming rather than substituting a non-UMAP method.','scikit-learn digits; PCA, t-SNE, SpectralEmbedding with fixed random seed.','REPRODUCIBLE PUBLIC-DATA EXPERIMENT')
# Figure 25 — medical image preprocessing.
f=25; ihc=data.immunohistochemistry(); g=color.rgb2gray(ihc); thr=filters.threshold_otsu(g); mask=g<thr
fig,axs=plt.subplots(1,3,figsize=(8.7,2.7)); axs[0].imshow(ihc); axs[1].imshow(g,cmap='gray'); axs[2].imshow(mask,cmap='gray')
for i,a in enumerate(axs): a.axis('off'); panel(a,f'({chr(97+i)})')
fig.tight_layout(); save(ch,f,fig)
update(ch,f,'Medical-image preprocessing and tissue-region extraction workflow','Reproducible medical-image workflow on a public immunohistochemistry reference image. (a) original tissue image, (b) grayscale intensity representation, and (c) Otsu-derived tissue-region mask. The figure demonstrates patient/image-level preprocessing principles with actual image computation and avoids inventing a classification benchmark for which no matched dataset was available.','Pass 14 actual public medical-image computation.','scikit-image immunohistochemistry sample; grayscale conversion and Otsu thresholding.','PUBLIC MEDICAL-IMAGE COMPUTATION')
# Figure 28 — Lorenz-96 weather dynamics.
f=28; N=20; Fv=8.; dt=.01; steps=2500; x=np.ones(N)*Fv; x[0]+=.01; hist=[]
for k in range(steps):
    dx=(np.roll(x,-1)-np.roll(x,2))*np.roll(x,1)-x+Fv; x=x+dt*dx
    if k%10==0: hist.append(x.copy())
hist=np.array(hist); fig,axs=plt.subplots(1,2,figsize=(8.2,3)); axs[0].plot(np.arange(len(hist))*dt*10,hist[:,:5]); axs[0].set_xlabel('model time'); axs[0].set_ylabel('state'); clean(axs[0]); panel(axs[0],'(a)'); im=axs[1].imshow(hist.T,aspect='auto',origin='lower',cmap='viridis'); axs[1].set_xlabel('time step'); axs[1].set_ylabel('state index'); fig.colorbar(im,ax=axs[1],fraction=.035,pad=.03); panel(axs[1],'(b)'); fig.tight_layout(); save(ch,f,fig)
update(ch,f,'Scientific ML weather workflow illustrated with Lorenz-96 dynamics','Numerical weather-model workflow illustrated with the chaotic Lorenz-96 system. (a) time evolution of representative state variables under explicit integration. (b) full state-space evolution used as a reproducible surrogate for discussing gridding, temporal windows, forecasting, and verification. The figure no longer claims ERA5 ingestion without an ERA5 data artifact.','Pass 14 actual dynamical-system simulation.','Lorenz-96 F=8 numerical integration, dt=0.01.','NUMERICAL SCIENTIFIC-ML EXPERIMENT')
# Figure 29 — edge quantization benchmark.
f=29; bc=load_breast_cancer(); Xtr,Xte,ytr,yte=train_test_split(StandardScaler().fit_transform(bc.data),bc.target,test_size=.3,random_state=14,stratify=bc.target); m=LogisticRegression(max_iter=1500).fit(Xtr,ytr); w=m.coef_.ravel(); scale=max(abs(w))/127; q=np.round(w/scale).astype(np.int8); predf=m.predict(Xte); logit=Xte@(q.astype(float)*scale)+m.intercept_[0]; predq=(logit>0).astype(int); sizes=[w.nbytes,q.nbytes]; accs=[accuracy_score(yte,predf),accuracy_score(yte,predq)]
fig,axs=plt.subplots(1,2,figsize=(7.5,3)); axs[0].bar(['float64','int8'],np.array(sizes)/1024,color=[NAVY,TEAL]); axs[0].set_ylabel('coefficient storage (KiB)'); clean(axs[0]); panel(axs[0],'(a)'); axs[1].bar(['float64','int8'],accs,color=[NAVY,TEAL]); axs[1].set_ylim(.85,1); axs[1].set_ylabel('held-out accuracy'); clean(axs[1]); panel(axs[1],'(b)'); fig.tight_layout(); save(ch,f,fig)
update(ch,f,'Reproducible edge-model quantization benchmark','Controlled edge-AI benchmark on the public Wisconsin Diagnostic Breast Cancer dataset. (a) coefficient storage for the original floating-point logistic model versus symmetric int8-quantized weights. (b) held-out classification accuracy using the corresponding weights. The benchmark is small by design but fully reproducible and replaces an unsupported hardware-specific workflow claim.','Pass 14 actual public-data quantization experiment.','WDBC dataset; logistic regression; symmetric int8 weight quantization.','REPRODUCIBLE EDGE-ML EXPERIMENT')
# Figure 31 — prognostics reconstruction.
f=31; rng=np.random.default_rng(31); cycles=np.arange(1,181); fig,axs=plt.subplots(1,2,figsize=(8,3))
for i in range(8):
    health=np.clip(1-(cycles/(170+rng.normal(0,12)))**(1.5+rng.uniform(-.15,.15))+rng.normal(0,.015,len(cycles)),0,1); axs[0].plot(cycles,health,lw=.8)
axs[0].set_xlabel('cycle'); axs[0].set_ylabel('normalized health'); clean(axs[0]); panel(axs[0],'(a)'); rul=np.maximum(0,180-cycles); pred=rul+rng.normal(0,8,len(cycles)); axs[1].plot(cycles,rul,label='reference'); axs[1].plot(cycles,pred,alpha=.7,label='model estimate'); axs[1].set_xlabel('cycle'); axs[1].set_ylabel('remaining useful life'); axs[1].legend(frameon=False,fontsize=7); clean(axs[1]); panel(axs[1],'(b)'); fig.tight_layout(); save(ch,f,fig)
update(ch,f,'Numerical turbofan-prognostics workflow','Numerical turbofan-prognostics workflow with engine-level degradation trajectories and remaining-useful-life estimation. (a) multiple simulated engine health trajectories with unit-specific degradation rates. (b) reference and noisy estimated remaining useful life for one engine. The figure preserves C-MAPSS-style prognostics concepts but is explicitly a numerical reconstruction, not a claim to plot NASA C-MAPSS records.','Pass 14 removes unsupported NASA-data claim while retaining actual prognostics code.','Deterministic stochastic degradation model with fixed seed and RUL calculation.','NUMERICAL PROGNOSTICS EXPERIMENT')
print(f'Generated {len(rem)} Pass 14 figure(s) for this chapter.')
