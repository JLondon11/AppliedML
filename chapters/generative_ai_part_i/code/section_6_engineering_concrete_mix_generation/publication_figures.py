"""Publication figures from real concrete strength results and generated-design outputs.
Requires results.json produced by node run.mjs Concrete_Data_Yeh.csv results.json.
"""
from pathlib import Path
import json,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
results=json.loads((root/'results.json').read_text())
test=pd.DataFrame(results['heldout']);design=pd.DataFrame(results['designrows'])
assert len(test)==229 and len(design)==200
out=root/'figures';out.mkdir(exist_ok=True)
navy='#1F587A';rust='#B46C4F';gray='#677B89';teal='#35796E'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'axes.edgecolor':'#A7B3BC','pdf.fonttype':42})
ordered=test.sort_values('strength_predicted')
groups=np.array_split(ordered,10)
bins=pd.DataFrame([dict(bin=i+1,n=len(p),real_mean=p.strength_measured.mean(),
  pred_mean=p.strength_predicted.mean(),real_sd=p.strength_measured.std(ddof=0)) for i,p in enumerate(groups)])
assert bins.n.sum()==229
fig,ax=plt.subplots(figsize=(7.8,4.5),layout='constrained')
ax.errorbar(bins.bin-.1,bins.real_mean,yerr=bins.real_sd,fmt='o-',capsize=3,markersize=5,lw=1.6,color=navy,label='Measured held-out strength (± within-decile SD)')
ax.plot(bins.bin+.1,bins.pred_mean,'D--',markersize=4.5,color=rust,label='Predicted strength, grouped by model ranking')
ax.set(xlabel='Held-out mixes ranked by model prediction (10 bins; 22–23 each)',
       ylabel='Concrete compressive strength (MPa)',xticks=np.arange(1,11))
ax.grid(axis='y',alpha=.18);ax.spines[['top','right']].set_visible(False)
ax.legend(frameon=False,fontsize=8.4,loc='upper left')
rmse=results['test_knn']['rmse'];baseline=results['test_mean_baseline']['rmse']
ax.text(.98,.06,f'Test RMSE {rmse:.2f} MPa  |  training-mean baseline {baseline:.2f} MPa',
        ha='right',transform=ax.transAxes,color='#365269',fontsize=8)
for ext in ('pdf','png'):fig.savefig(out/('concrete_measured_holdout.'+ext),dpi=230,bbox_inches='tight')
plt.close(fig)
design['cement_bin_center']=np.floor(design.cement/50)*50+25
fig,ax=plt.subplots(figsize=(7.9,4.5),layout='constrained')
for arm,color,mark,label in [('unconditioned_training_mixes',gray,'o','Unconditional empirical kernel'),
 ('high_strength_conditioned',teal,'D','Conditioned on high-strength real mixes')]:
 p=design[design.arm==arm].groupby('cement_bin_center').agg(n=('idx','size'),strength=('predicted_strength','mean')).reset_index()
 ax.plot(p.cement_bin_center,p.strength,color=color,lw=1.7,alpha=.9)
 ax.scatter(p.cement_bin_center,p.strength,s=17+p.n*4,color=color,marker=mark,edgecolors='white',linewidth=.7,label=label)
ax.axhline(40,ls=':',color=rust,lw=1.2,label='Surrogate screening target (40 MPa)')
ax.set(xlabel='Generated cement mass (kg/m³), 50-unit bins',ylabel='Surrogate-predicted strength (MPa)',xlim=(100,600),ylim=(18,74))
ax.grid(axis='y',alpha=.16);ax.spines[['top','right']].set_visible(False)
ax.legend(frameon=False,fontsize=8,loc='upper left')
ax.text(.98,.05,'Marker area proportional to count; first 100 saved candidates per arm',
        ha='right',transform=ax.transAxes,fontsize=8,color='#405664')
for ext in ('pdf','png'):fig.savefig(out/('concrete_generated_candidates.'+ext),dpi=230,bbox_inches='tight')
plt.close(fig)
print('PASS: 229 genuine held-out laboratory records, 200 recorded generation outputs')
