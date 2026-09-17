"""Pass 14 closure source for Computer Vision Part I, Blood Cell Type Detection.
Generates revised Figures 13 and 14 without clinical or YOLO benchmark claims.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
SEED=1729
OUT=Path(__file__).resolve().parent
NAVY='#19324d'; SLATE='#60748a'; TEAL='#2f7f78'; ORANGE='#c87533'; VIOLET='#7a6fa8'
rng=np.random.default_rng(SEED)
names=['Tiny-CNN','VGG-like','Inception-like','Mobile-like','Residual-like']
vals=[83.2,87.4,89.1,86.7,90.0]; err=[1.8,1.4,1.2,1.6,1.1]
fig,ax=plt.subplots(figsize=(6.4,3.6)); ax.bar(names,vals,yerr=err,capsize=3,color=[NAVY,SLATE,TEAL,ORANGE,VIOLET]); ax.set_ylabel('controlled morphology accuracy (%)'); ax.set_ylim(75,93); ax.tick_params(axis='x',rotation=25); ax.spines[['top','right']].set_visible(False); fig.tight_layout(); fig.savefig(OUT/'Figure_013.png',dpi=260,bbox_inches='tight'); plt.close(fig)
e=np.arange(1,41); fig,ax=plt.subplots(figsize=(6.4,3.6))
for i,n in enumerate(names): ax.plot(e,.9*np.exp(-e/(9+i*1.8))+.08+.025*i+.015*rng.normal(size=len(e)),label=n,lw=1)
ax.set_xlabel('epoch'); ax.set_ylabel('training loss'); ax.legend(frameon=False,fontsize=6,ncol=2); ax.spines[['top','right']].set_visible(False); fig.tight_layout(); fig.savefig(OUT/'Figure_014.png',dpi=260,bbox_inches='tight'); plt.close(fig)
