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
rem=[]; ch='09_NLP_Part_II'
ks=np.array([1,2,3,5,10,20,50,100]); rec=np.array([.39944,.50233,.56067,.61244,.693,.78,.84622,.87356]); irr=np.array([.58667,.73167,.79778,.86533,.923,.95617,.98087,.9901]); prec=1-irr
# Figure 14 — evidence precision/recall tradeoff.
f=14; fig,ax=plt.subplots(figsize=(6.4,3.2)); ax.plot(ks,prec,marker='o',label='evidence precision'); ax.plot(ks,rec,marker='s',label='evidence recall'); ax.set_xscale('log'); ax.set_xlabel('retrieved depth k'); ax.set_ylabel('fraction'); ax.set_ylim(0,1); ax.legend(frameon=False); clean(ax); fig.tight_layout(); save(ch,f,fig)
update(ch,f,'SciFact evidence precision–recall tradeoff for retrieval augmentation','Measured evidence precision and recall as retrieval depth increases in the repository-backed SciFact retrieval experiment. Precision is computed as one minus the recorded irrelevant-document fraction, while recall is the recorded mean evidence recall. The figure replaces an unsupported generic citation-accuracy improvement claim with metrics derived from the actual ranked retrieval output.','Pass 14 derived directly from stored Pass13 SciFact metrics.','BEIR SciFact TF-IDF retrieval experiment; precision=1-irrelevant fraction, recorded recall@k.','REPRODUCIBLE RETRIEVAL METRIC')
# Figures 15 and 18 — controlled unsupported-content experiment.
for f,title in [(15,'Scientific unsupported-content reduction under retrieval augmentation'),(18,'Unsupported-response comparison in controlled RAG')]:
    vals=[.51524,.46340]; lo=[.47035,.39347]; hi=[.56021,.53070]; fig,ax=plt.subplots(figsize=(5.8,3.2)); ax.bar(['standalone','retrieval-grounded'],vals,yerr=np.array([[vals[0]-lo[0],vals[1]-lo[1]],[hi[0]-vals[0],hi[1]-vals[1]]]),capsize=4,color=[NAVY,TEAL]); ax.set_ylabel('unsupported-term fraction'); ax.set_ylim(0,.7); clean(ax); fig.tight_layout(); save(ch,f,fig)
    update(ch,f,title,'Controlled SciFact generation comparison using the repository-backed experiment. Bars show mean unsupported-term fraction for standalone generation and retrieval-grounded generation; error bars show the stored bootstrap confidence interval. The figure is explicitly tied to this protocol rather than generalized as a universal hallucination rate.','Pass 14 exact reuse of repository-backed controlled-generation metrics.','SciFact + FLAN-T5-small controlled generation; stored mean unsupported-term fractions and bootstrap intervals.','REPRODUCIBLE RAG EXPERIMENT')
# Figure 17 — retrieval depth strata.
f=17; groups=['shallow\n(k≤3)','medium\n(k=5–20)','deep\n(k≥50)']; vals=[rec[:3].mean(),rec[3:6].mean(),rec[6:].mean()]; fig,ax=plt.subplots(figsize=(5.8,3.1)); ax.bar(groups,vals,color=[NAVY,TEAL,ORANGE]); ax.set_ylabel('mean evidence recall'); ax.set_ylim(0,1); clean(ax); fig.tight_layout(); save(ch,f,fig)
update(ch,f,'Retrieval quality across SciFact retrieval-depth strata','Measured retrieval quality summarized across three retrieval-depth strata from the same SciFact experiment: shallow (k≤3), medium (k=5–20), and deep (k≥50). Bars average the recorded mean evidence recall within each stratum. This replaces the unsupported enterprise-document-category claim with a directly traceable stratified analysis.','Pass 14 derived from stored SciFact retrieval curve.','Repository-backed SciFact recall@k metrics grouped by retrieval depth.','REPRODUCIBLE RETRIEVAL METRIC')
# Figure 21 — architecture capability matrix.
f=21; M=np.array([[1,.4,.3,.5],[.8,.8,.6,.7],[.7,.9,.9,.8],[.6,.8,1,.9]]); fig,ax=plt.subplots(figsize=(6.6,3.1)); im=ax.imshow(M,vmin=0,vmax=1,cmap='viridis',aspect='auto'); ax.set_yticks(range(4),['OCR+classifier','layout model','vision-language','retrieval-augmented']); ax.set_xticks(range(4),['text','layout','vision','grounding']); fig.colorbar(im,ax=ax,fraction=.035,pad=.03,label='normalized capability'); fig.tight_layout(); save(ch,f,fig)
update(ch,f,'Document-intelligence architecture capability matrix','Normalized capability matrix comparing four document-intelligence architecture families across text extraction, layout modeling, visual reasoning, and evidence grounding. The values encode explicit architectural capabilities on a 0–1 design scale and are not misrepresented as benchmark scores.','Pass 14 replaces unsupported benchmark values with an explicit design-capability matrix.','Deterministic architecture-capability encoding; no empirical benchmark claim.','SCIENTIFIC METHOD MATRIX')
# Figures 24 and 25 — evidence coverage and precision.
for f,metric,title in [(24,rec,'Evidence coverage across retrieval configurations'),(25,prec,'Citation precision across retrieval configurations')]:
    sel=[1,3,10,50]; idxs=[list(ks).index(k) for k in sel]; vals=[metric[i] for i in idxs]; fig,ax=plt.subplots(figsize=(6.1,3.1)); ax.bar([f'k={k}' for k in sel],vals,color=COLORS[:4]); ax.set_ylim(0,1); ax.set_ylabel('evidence recall' if f==24 else 'evidence precision'); clean(ax); fig.tight_layout(); save(ch,f,fig)
    cap=('Measured evidence coverage for four retrieval configurations from the repository-backed SciFact experiment. Bars report recorded mean evidence recall at k=1, 3, 10, and 50, exposing the gain in coverage as more evidence is retrieved.' if f==24 else 'Measured citation/evidence precision for four retrieval configurations from the repository-backed SciFact experiment. Precision is one minus the recorded irrelevant-document fraction at k=1, 3, 10, and 50; the plot makes the precision–coverage tradeoff explicit.')
    update(ch,f,title,cap,'Pass 14 exact derivation from SciFact retrieval metrics.','Repository-backed SciFact retrieval metrics.','REPRODUCIBLE RETRIEVAL METRIC')
print(f'Generated {len(rem)} Pass 14 figure(s) for this chapter.')
