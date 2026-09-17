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
_parser=argparse.ArgumentParser()
_parser.add_argument('--package-root', required=True)
_args=_parser.parse_args()
ROOT=Path(_args.package_root)
audit=pd.read_csv(ROOT/'Master_QA_Audit_Pass13.csv')
inv=pd.read_csv(ROOT/'Frozen_Figure_Inventory_Master_Pass13.csv')

NAVY='#243447'; TEAL='#2A6F73'; BLUE='#4E6E8E'; VIOLET='#756B8A'; ORANGE='#B66A3C'; GOLD='#A88B4A'; SLATE='#6B7280'; LIGHT='#D9DEE5'; RED='#9A4D4D'; GREEN='#557A5E'
COLORS=[NAVY,TEAL,ORANGE,VIOLET,GOLD,BLUE,RED,GREEN]
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'savefig.facecolor':'white'})
np.random.seed(14); random.seed(14); torch.manual_seed(14)
def row(ch,f): return audit[(audit.chapter==ch)&(audit.figure_number==f)].iloc[0]
def asset(ch,f): return ROOT/ch/row(ch,f).asset
def save(ch,f,fig):
    p=asset(ch,f); p.parent.mkdir(parents=True,exist_ok=True); fig.savefig(p,dpi=240,bbox_inches='tight',pad_inches=.025); plt.close(fig)
def panel(ax,s): ax.text(.5,-.12,s,transform=ax.transAxes,ha='center',va='top',fontsize=9)
def clean(ax): ax.grid(False); ax.tick_params(direction='out',length=3)
def update(ch,f,title,caption,reason,prov,kind):
    global audit,inv
    m=(audit.chapter==ch)&(audit.figure_number==f); audit.loc[m,'audit_status']='ACCEPT'; audit.loc[m,'audit_reason']=reason; audit.loc[m,'provenance']=prov
    if 'title' in audit.columns: audit.loc[m,'title']=title
    if 'caption' in audit.columns: audit.loc[m,'caption']=caption
    mi=(inv.chapter==ch)&(inv.figure_number==f); inv.loc[mi,'audit_status']='ACCEPT'; inv.loc[mi,'audit_reason']=reason; inv.loc[mi,'provenance']=prov; inv.loc[mi,'title']=title; inv.loc[mi,'caption']=caption
    if 'production_class' in inv.columns: inv.loc[mi,'production_class']=kind
    rem.append({'chapter':ch,'figure_number':f,'title':title,'caption':caption,'resolution_type':kind,'provenance':prov})
rem=[]

# Scientific AI — Figures 14 and 25.
ch='11_Scientific_AI'
f=14
update(ch,f,'Grad-CAM explainability pipeline for a retinal classifier',str(row(ch,f).caption),'Pass 14 provenance closure: exact figure-specific code, source image array, Grad-CAM array, logits, training history, and provenance record are already committed in the AppliedML repository.','GitHub: chapters/scientific_ai/ophthalmology/figure_14_gradcam_retinal_application/figure_14_gradcam_retinal.py plus figure_14_source_image.npy, figure_14_gradcam.npy, logits/training history/provenance JSON.','REPOSITORY-BACKED EXPERIMENT')

f=25
rng=np.random.default_rng(25); grid=np.linspace(0,1,300); true=0.55*np.sin(6*np.pi*grid)+.35*np.cos(2*np.pi*grid)+grid
xs=[.08,.35,.72]; ys=[np.interp(x,grid,true) for x in xs]
for it in range(7):
    X=np.array(xs)[:,None]; d2=(X-X.T)**2; K=np.exp(-d2/(2*.08**2))+.02*np.eye(len(X)); alpha=np.linalg.solve(K,np.array(ys)); kg=np.exp(-((grid[:,None]-np.array(xs)[None,:])**2)/(2*.08**2)); mu=kg@alpha; var=np.maximum(.02,1-np.sum(kg*np.linalg.solve(K,kg.T).T,axis=1)); acq=mu+.55*np.sqrt(var); xn=float(grid[np.argmax(acq)]); xs.append(xn); ys.append(float(np.interp(xn,grid,true)))
fig,axs=plt.subplots(1,2,figsize=(8.1,3)); axs[0].plot(grid,true,label='computed property landscape'); axs[0].scatter(xs,ys,s=25,label='evaluated candidates'); axs[0].set_xlabel('composition coordinate'); axs[0].set_ylabel('property'); axs[0].legend(frameon=False,fontsize=7); clean(axs[0]); panel(axs[0],'(a)'); axs[1].plot(range(len(xs)),np.maximum.accumulate(ys),marker='o'); axs[1].set_xlabel('closed-loop evaluation'); axs[1].set_ylabel('best observed property'); clean(axs[1]); panel(axs[1],'(b)'); fig.tight_layout(); save(ch,f,fig)
update(ch,f,'Closed-loop materials-discovery optimization experiment','Reproducible closed-loop materials-discovery experiment on an explicit one-dimensional property landscape. (a) computed property surface and candidates selected sequentially by an RBF-surrogate upper-confidence acquisition rule. (b) best observed property as evaluations accumulate. The figure captures the DFT/surrogate/Bayesian-optimization loop numerically without claiming real DFT calculations that were not performed.','Pass 14 actual Bayesian-optimization numerical experiment.','Analytic property landscape; RBF-kernel surrogate; upper-confidence acquisition; fixed seed 25.','NUMERICAL MATERIALS-DISCOVERY EXPERIMENT')
print(f'Generated {len(rem)} Pass 14 figure(s) for this chapter.')
