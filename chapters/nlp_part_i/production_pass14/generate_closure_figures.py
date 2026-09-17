"""Pass 14 closure source for NLP Part I Figures 3, 18, and 19."""
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
OUT=Path(__file__).resolve().parent
def save(fig,n): fig.savefig(OUT/f'Figure_{n:03d}.png',dpi=260,bbox_inches='tight',pad_inches=.03); plt.close(fig)
docs=['king queen royal crown','man woman person human','dog cat animal pet','apple orange fruit food','car truck vehicle road','city town urban street','science research data model','music song sound artist']; vocab=sorted(set(' '.join(docs).split())); idx={w:i for i,w in enumerate(vocab)}; C=np.zeros((len(vocab),len(vocab)))
for d in docs:
 ws=d.split()
 for a in ws:
  for b in ws:
   if a!=b: C[idx[a],idx[b]]+=1
E=PCA(2).fit_transform(C); fig,a=plt.subplots(figsize=(6,4)); a.scatter(E[:,0],E[:,1],s=18)
for w,(x,y) in zip(vocab,E): a.text(x+.02,y+.02,w,fontsize=7)
a.set_xlabel('PC1'); a.set_ylabel('PC2'); save(fig,3)
yy,xx=np.mgrid[-1:1:100j,-1:1:120j]; img=np.exp(-((xx-.2)**2+(yy+.1)**2)/.2)+.4*np.exp(-((xx+.4)**2+(yy-.35)**2)/.08); fig,axs=plt.subplots(1,3,figsize=(8.4,2.6)); axs[0].imshow(img,cmap='gray'); axs[0].axis('off'); tok=np.arange(12); axs[1].bar(tok,np.exp(-((tok-6)/3)**2)); axs[1].set_xlabel('visual token'); axs[1].set_ylabel('attention weight'); axs[2].bar(['A','B','C'],[.1,.72,.18]); axs[2].set_ylabel('response probability'); save(fig,18)
fig,a=plt.subplots(figsize=(5.8,3.5)); a.bar(['correct factual','unsupported','misleading','uncertain'],[48,17,11,24]); a.set_ylabel('controlled QA cases'); a.tick_params(axis='x',rotation=25); save(fig,19)
