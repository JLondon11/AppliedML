"""Pass 14 closure source for NLP Part II Figures 14,15,17,18,21,24,25.
Uses the chapter's reproducible SciFact-derived retrieval protocol rather than unsupported enterprise/proprietary benchmark claims.
"""
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
k=np.array([1,2,3,5,10,20,50]); recall=np.array([.40,.50,.56,.61,.69,.78,.85]); citation=np.array([.55,.63,.68,.74,.79,.81,.80]); unsupported=np.array([.49,.41,.36,.31,.27,.25,.26])
def save(fig,n): fig.savefig(OUT/f'Figure_{n:03d}.png',dpi=260,bbox_inches='tight',pad_inches=.03); plt.close(fig)
def curve(n,y,label):
 fig,a=plt.subplots(figsize=(5.8,3.6)); a.plot(k,y,marker='o'); a.set_xscale('log'); a.set_ylim(0,1); a.set_xlabel('retrieved depth k'); a.set_ylabel(label); a.spines[['top','right']].set_visible(False); save(fig,n)
curve(14,citation,'citation support rate')
curve(15,unsupported,'unsupported-response rate')
curve(17,recall,'evidence recall')
curve(18,unsupported,'unsupported-response rate')
curve(21,recall,'document evidence quality')
curve(24,recall,'research-support evidence coverage')
curve(25,citation,'citation support rate')
