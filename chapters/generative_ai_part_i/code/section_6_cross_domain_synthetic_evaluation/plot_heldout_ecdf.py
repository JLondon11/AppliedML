"""Chapter 6 real-data domain-gap ECDF figure from executed saved predictions.

Input: results/predictions.csv and results/metrics.csv, produced by run.py.
Output: results/cross_domain_heldout_ecdf.pdf/.svg/.png.
"""
import csv
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent/'results'
with (root/'predictions.csv').open(newline='') as f: predictions=list(csv.DictReader(f))
with (root/'metrics.csv').open(newline='') as f: metrics=list(csv.DictReader(f))
assert len(predictions)==2592 and len(metrics)==6
series={}
for seed in (20261106,20261107,20261108):
    for arm in ('real_source','gan_source'):
        rec=sorted((r for r in predictions if int(r['seed'])==seed and r['arm']==arm),key=lambda r:int(r['test_index']))
        assert len(rec)==432
        y=np.array([int(r['ground_truth_target_domain']) for r in rec])
        p=np.array([float(r['probability_target_domain']) for r in rec])
        assert np.array_equal(y,np.r_[np.zeros(360,dtype=int),np.ones(72,dtype=int)])
        m=next(r for r in metrics if int(r['seed'])==seed and r['arm']==arm)
        assert abs(balanced_accuracy_score(y,p>=0.5)-float(m['test_balanced_accuracy']))<1e-10
        assert abs(roc_auc_score(y,p)-float(m['test_roc_auc']))<1e-10
        series[seed,arm]=p
assert all(np.array_equal(series[20261106,'real_source'],series[s,'real_source']) for s in (20261107,20261108))
colors={'real':'#17637B',20261106:'#C47A53',20261107:'#8171A3',20261108:'#4C927C'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(11.2,4.25),layout='constrained',sharey=True)
for ax,(a,b,title) in zip(axs,[(0,360,'Real NEU steel-defect photographs (n = 360)'),(360,432,'Real brick / grass / gravel patches (n = 72)')]):
    for seed,arm,name,style in [(20261106,'real_source','Real-source-trained reference','-'),(20261106,'gan_source','GAN source · seed 20261106','--'),(20261107,'gan_source','GAN source · seed 20261107','--'),(20261108,'gan_source','GAN source · seed 20261108','--')]:
        p=np.sort(series[seed,arm][a:b]);n=len(p)
        ax.step(np.r_[0,p,1],np.r_[0,np.arange(1,n+1)/n,1],where='post',
                color=colors['real' if arm=='real_source' else seed],
                linewidth=2.2 if arm=='real_source' else 1.55,ls=style,label=name)
    ax.axvline(.5,color='#657989',ls=':',lw=1)
    ax.set(title=title,xlabel='Predicted probability of texture domain',xlim=(0,1),ylim=(0,1.02))
    ax.grid(color='#E5ECF0',lw=.7)
axs[0].set_ylabel('Empirical cumulative fraction of genuine held-out records')
axs[1].legend(loc='upper left',fontsize=8.2,frameon=False)
fig.suptitle('Domain fidelity: held-out real photographs under two source-training regimes',
             fontsize=12.2,fontweight='bold')
for ext in ('pdf','svg','png'):
    fig.savefig(root/('cross_domain_heldout_ecdf.'+ext),dpi=220,bbox_inches='tight')
plt.close(fig)
print('PASS: 2,592 predictions recomputed; vector and high-resolution real-data figures generated')
