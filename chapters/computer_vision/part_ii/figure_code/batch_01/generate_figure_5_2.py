"""Reproducible exact numerical demonstration of valid 2D cross-correlation.
Scientific figure 5.2; deterministic generated inputs, NOT an empirical benchmark.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT=Path(__file__).resolve().parents[1]/'figures'
OUT.mkdir(parents=True,exist_ok=True)
X=np.array([[2,1,0,1,3],[1,3,2,0,1],[0,2,1,4,2],[3,1,0,2,1],[1,2,3,1,0]],dtype=int)
K=np.array([[1,0,-1],[1,0,-1],[1,0,-1]],dtype=int)
Y=np.array([[np.sum(X[i:i+3,j:j+3]*K) for j in range(3)] for i in range(3)])
assert Y.shape==(3,3) and Y[0,0]==0
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=(11.6,4.3),facecolor='white')
axes=[fig.add_axes(p) for p in [(0.045,0.24,0.31,0.68),(0.435,0.34,0.18,0.48),(0.74,0.34,0.18,0.48)]]
labels=['Input $X$ (5 × 5)','Kernel $K$ (3 × 3)','Output $Y$ (3 × 3)']
colors=[('#F0F3F7','#B88837'),('#E5EFE9','#276A57'),('#E9EFF8','#1C537B')]
for ax,A,title,(fill,edge) in zip(axes,[X,K,Y],labels,colors):
  ax.imshow(np.zeros_like(A),cmap='gray',vmin=0,vmax=1)
  n,m=A.shape
  for i in range(n):
    for j in range(m):
      ax.add_patch(Rectangle((j-0.5,i-0.5),1,1,fc=fill,ec='#BEC5CB',lw=.8))
      ax.text(j,i,str(A[i,j]),ha='center',va='center',fontsize=13,color='#1E3147')
  ax.set_xlim(-.5,m-.5);ax.set_ylim(n-.5,-.5);ax.set_aspect('equal');ax.axis('off')
  ax.set_title(title,fontsize=13,color='#1E3147',pad=10)
axes[0].add_patch(Rectangle((-.5,-.5),3,3,fc='none',ec='#B88837',lw=3))
axes[2].add_patch(Rectangle((-.5,-.5),1,1,fc='none',ec='#1C537B',lw=3))
fig.add_artist(FancyArrowPatch((.365,.57),(.421,.57),transform=fig.transFigure,arrowstyle='-|>',mutation_scale=20,lw=1.4,color='#596D7F'))
fig.add_artist(FancyArrowPatch((.624,.57),(.722,.57),transform=fig.transFigure,arrowstyle='-|>',mutation_scale=20,lw=1.4,color='#596D7F'))
fig.text(.5,.115,r'$Y_{ij}=\sum_{a=0}^{2}\sum_{b=0}^{2}K_{ab}\,X_{i+a,j+b}\quad (i,j=0,1,2)$',ha='center',fontsize=15,color='#20364A')
fig.text(.5,.035,'Valid cross-correlation · unit stride · no padding · exact integer arithmetic',ha='center',fontsize=9,color='#526779')
for ext in ('png','svg','pdf'):
 fig.savefig(OUT/f'Figure_5_02_Convolution.{ext}',dpi=240,bbox_inches='tight',pad_inches=.13)
plt.close(fig)
np.savetxt(OUT/'Figure_5_02_input_matrix.csv',X,delimiter=',',fmt='%d')
np.savetxt(OUT/'Figure_5_02_kernel.csv',K,delimiter=',',fmt='%d')
np.savetxt(OUT/'Figure_5_02_output.csv',Y,delimiter=',',fmt='%d')
print(Y)
