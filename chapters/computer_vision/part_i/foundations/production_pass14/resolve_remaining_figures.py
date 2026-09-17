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

# premium restrained palette
NAVY='#243447'; TEAL='#2A6F73'; BLUE='#4E6E8E'; VIOLET='#756B8A'; ORANGE='#B66A3C'; GOLD='#A88B4A'; SLATE='#6B7280'; LIGHT='#D9DEE5'; RED='#9A4D4D'; GREEN='#557A5E'
COLORS=[NAVY,TEAL,ORANGE,VIOLET,GOLD,BLUE,RED,GREEN]
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'savefig.facecolor':'white'})
np.random.seed(14); random.seed(14); torch.manual_seed(14)

def row(ch,f): return audit[(audit.chapter==ch)&(audit.figure_number==f)].iloc[0]
def asset(ch,f): return ROOT/ch/row(ch,f).asset
def save(ch,f,fig):
    p=asset(ch,f); p.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(p,dpi=240,bbox_inches='tight',pad_inches=.025)
    plt.close(fig)
def panel(ax,s): ax.text(.5,-.12,s,transform=ax.transAxes,ha='center',va='top',fontsize=9)
def clean(ax): ax.grid(False); ax.tick_params(direction='out',length=3)

def update(ch,f,title,caption,reason,prov,kind):
    global audit,inv
    m=(audit.chapter==ch)&(audit.figure_number==f)
    audit.loc[m,'audit_status']='ACCEPT'; audit.loc[m,'audit_reason']=reason; audit.loc[m,'provenance']=prov
    if 'title' in audit.columns: audit.loc[m,'title']=title
    if 'caption' in audit.columns: audit.loc[m,'caption']=caption
    mi=(inv.chapter==ch)&(inv.figure_number==f)
    inv.loc[mi,'audit_status']='ACCEPT'; inv.loc[mi,'audit_reason']=reason; inv.loc[mi,'provenance']=prov
    inv.loc[mi,'title']=title; inv.loc[mi,'caption']=caption
    if 'production_class' in inv.columns: inv.loc[mi,'production_class']=kind
    rem.append({'chapter':ch,'figure_number':f,'title':title,'caption':caption,'resolution_type':kind,'provenance':prov})

rem=[]

# Figures 13-14 — Application: Blood Cell Type Detection.
ch='01_Computer_Vision_Part_I_Foundations'
models=['Tiny YOLO','VGG-16','ResNet50','InceptionV3','MobileNet']
rbc=[96.09,72.98,79.80,87.75,74.24]; wbc=[86.89,100,95.08,100,93.44]; platelet=[96.36,90.91,87.27,96.36,83.64]
maps=[62.36,71.32,74.37,68.26,52.07]; runtime=[60,106,118,130,84]

f=13
fig,axs=plt.subplots(1,2,figsize=(8.6,3.1)); x=np.arange(len(models)); width=.23
for vals,off,label,c in [(rbc,-width,'RBC',NAVY),(wbc,0,'WBC',TEAL),(platelet,width,'Platelet',ORANGE)]: axs[0].bar(x+off,vals,width,label=label,color=c)
axs[0].set_xticks(x,models,rotation=25,ha='right'); axs[0].set_ylabel('Counting accuracy (%)'); axs[0].set_ylim(65,103); axs[0].legend(frameon=False,ncol=3,fontsize=7); clean(axs[0]); panel(axs[0],'(a)')
axs[1].barh(models,maps,color=COLORS[:5]); axs[1].set_xlabel('Validation mAP (%)'); axs[1].set_xlim(45,80); clean(axs[1]); panel(axs[1],'(b)')
fig.tight_layout(); save(ch,f,fig)
update(ch,f,'Published blood-cell detection performance across CNN backbones','Published blood-cell detection performance for Tiny YOLO, VGG-16, ResNet50, InceptionV3, and MobileNet used as the YOLO backbone. (a) Test-set counting accuracy for red blood cells, white blood cells, and platelets. (b) Validation mean average precision (mAP). Values reproduce Table 3 of Alam and Islam (2019), using their modified BCCD protocol; the figure does not mix these values with results from a different split or implementation.','Pass 14 source-backed reproduction from published numeric table.','Alam & Islam 2019 Healthcare Technology Letters Table 3; open-access CC BY 3.0; published RBC/WBC/platelet accuracy and mAP values.','PUBLISHED-DATA REPRODUCTION')

f=14
fig,axs=plt.subplots(1,2,figsize=(8.5,3.0))
axs[0].bar(models,maps,color=COLORS[:5]); axs[0].set_ylabel('Validation mAP (%)'); axs[0].tick_params(axis='x',rotation=25); clean(axs[0]); panel(axs[0],'(a)')
axs[1].scatter(runtime,maps,s=55,color=COLORS[:5])
for a,b,n in zip(runtime,maps,models): axs[1].annotate(n,(a,b),xytext=(4,3),textcoords='offset points',fontsize=7)
axs[1].set_xlabel('Execution time per test image (ms)'); axs[1].set_ylabel('Validation mAP (%)'); clean(axs[1]); panel(axs[1],'(b)')
fig.tight_layout(); save(ch,f,fig)
update(ch,f,'Published CNN–YOLO accuracy and inference-time comparison','Published CNN-backbone comparison for the blood-cell YOLO experiment of Alam and Islam (2019). (a) Validation mAP for Tiny YOLO, VGG-16, ResNet50, InceptionV3, and MobileNet. (b) Accuracy–latency tradeoff using the authors’ reported average execution time per test image. This source-backed replacement avoids inventing pointwise training-loss values that were not available numerically from the paper.','Pass 14 replaces an unverifiable curve recreation with directly published quantitative evidence.','Alam & Islam 2019 Table 3: mAP and execution time per test image.','PUBLISHED-DATA REPRODUCTION')

print(f'Generated {len(rem)} Pass 14 figure(s) for this chapter.')
