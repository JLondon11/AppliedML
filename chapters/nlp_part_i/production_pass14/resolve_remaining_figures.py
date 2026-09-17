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
rem=[]
ch='08_NLP_Part_I'
# Figure 3 — word embeddings.
f=3
sentences=['king queen royal palace','queen princess royal crown','dog cat animal pet','cat kitten animal pet','car truck road vehicle','bus car road vehicle','apple orange fruit food','banana apple fruit food','doctor nurse hospital health','teacher student school education']
words=sorted(set(' '.join(sentences).split())); idx={w:i for i,w in enumerate(words)}; C=np.zeros((len(words),len(words)))
for s in sentences:
    toks=s.split()
    for w in toks:
        for v in toks:
            if v!=w: C[idx[w],idx[v]]+=1
E=PCA(2).fit_transform(C); fig,ax=plt.subplots(figsize=(6.4,4)); ax.scatter(E[:,0],E[:,1],s=28,color=TEAL)
for w,(x0,y0) in zip(words,E): ax.text(x0+.03,y0+.03,w,fontsize=7)
ax.set_xlabel('embedding component 1'); ax.set_ylabel('embedding component 2'); clean(ax); fig.tight_layout(); save(ch,f,fig)
update(ch,f,'Distributional word-embedding projection','Two-dimensional projection of word vectors computed from an explicit co-occurrence matrix over a small controlled corpus. Semantically related words occupy nearby regions because they share contexts. The figure reproduces the pedagogical role of an embedding-projector visualization without falsely claiming that the coordinates were exported from TensorFlow Embedding Projector.','Pass 14 exact co-occurrence/PCA computation.','Controlled corpus; word-context co-occurrence matrix; PCA projection.','CODED NLP EMBEDDING EXPERIMENT')
# Figure 18 — multimodal prompting schematic.
f=18
fig,axs=plt.subplots(1,2,figsize=(7.8,3.0)); axs[0].imshow(data.chelsea()); axs[0].axis('off'); axs[0].text(.02,.02,'visual input',transform=axs[0].transAxes,fontsize=8,bbox=dict(facecolor='white',alpha=.8,edgecolor='none')); axs[1].axis('off'); txt='Prompt: Describe the visible animal and scene.\n\nModel response structure:\n• identifies salient visual objects\n• describes color/pose/context\n• states uncertainty when details are ambiguous'; axs[1].text(.02,.95,txt,va='top',fontsize=9,linespacing=1.5); panel(axs[0],'(a)'); panel(axs[1],'(b)'); fig.tight_layout(); save(ch,f,fig)
update(ch,f,'Multimodal visual-input prompting schematic','Scientific schematic of multimodal prompting. (a) a public reference image supplied as visual input. (b) the corresponding prompt/response structure expected from a vision-language model. The figure demonstrates the visual-input capability concept without presenting a fabricated screenshot or attributing a generated response to a specific unavailable GPT-4 session.','Pass 14 non-empirical code-rendered multimodal schematic.','Public scikit-image Chelsea image and code-rendered prompt/response structure.','NON-EMPIRICAL SCIENTIFIC RENDERING')
# Figure 19 — TruthfulQA structure.
f=19
fig,ax=plt.subplots(figsize=(8.0,3.4)); ax.axis('off'); rows=[['Question type','Truthful reference','Common misconception'],['Health myth','Evidence-based correction','Popular but unsupported claim'],['Folk belief','Qualified factual answer','Confident false premise']]; table=ax.table(cellText=rows[1:],colLabels=rows[0],loc='center',cellLoc='left'); table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1,1.7); fig.tight_layout(); save(ch,f,fig)
update(ch,f,'TruthfulQA correct-versus-misconception response structure','Compact response-analysis table illustrating the TruthfulQA evaluation distinction between a truthful reference answer and a plausible misconception for two question types. The production figure no longer fabricates specific GPT-4 answers; GPT-4 TruthfulQA results remain attributed to the cited OpenAI report in the surrounding text.','Pass 14 removes unsupported generated-answer attribution.','Code-rendered evaluation structure based on TruthfulQA task definition; no fabricated model outputs.','BENCHMARK-STRUCTURE RENDERING')
print(f'Generated {len(rem)} Pass 14 figure(s) for this chapter.')
