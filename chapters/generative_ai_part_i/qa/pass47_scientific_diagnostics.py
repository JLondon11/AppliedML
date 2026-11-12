"""Chapter 6 data-backed supplementary science figures (Pass47).

Reads actual saved Mauna Loa CVAE and Wisconsin cytology held-out predictions.
No synthetic numbers. No claims of independent test subjects across model seeds.
"""
import csv,json,argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
p=argparse.ArgumentParser()
p.add_argument('--evidence-root',type=Path,required=True)
p.add_argument('--output',type=Path,default=Path('pass47_figures'))
args=p.parse_args()
root=args.evidence_root;out=args.output;out.mkdir(parents=True,exist_ok=True)
def load(path):
 with path.open(newline='') as f:return list(csv.DictReader(f))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,
 'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#81909B',
 'axes.labelcolor':'#213442','xtick.color':'#536777','ytick.color':'#536777'})
colors=['#165F73','#BD7554','#4D8B7D']
real=load(root/'pass19/Chapter6_Pass19/Climate_CO2_CVAE/results/test_predictions.csv')
fig,axes=plt.subplots(1,2,figsize=(10.5,4.25),layout='constrained')
diagnostics={}
for seed,color in zip([17,23,41],colors):
 sub=[r for r in real if int(r['seed'])==seed]
 coverage=[];rmse=[]
 for lead in range(1,9):
  q=[r for r in sub if int(r['horizon_week'])==lead]
  assert len(q)==306
  actual=np.array([float(r['true_co2_ppm']) for r in q])
  lo=np.array([float(r['cvae_q05_ppm']) for r in q])
  hi=np.array([float(r['cvae_q95_ppm']) for r in q])
  mean=np.array([float(r['cvae_mean_ppm']) for r in q])
  coverage.append(float(np.mean((actual>=lo)&(actual<=hi))*100))
  rmse.append(float(np.sqrt(np.mean((actual-mean)**2))))
 diagnostics[seed]={'coverage_by_lead_pct':coverage,'RMSE_by_lead_ppm':rmse}
 axes[0].plot(range(1,9),coverage,marker='o',color=color,linewidth=1.5,markersize=4,label=f'Seed {seed}')
 axes[1].plot(range(1,9),rmse,marker='o',color=color,linewidth=1.5,markersize=4)
axes[0].axhline(90,color='#798B98',linewidth=1.2,ls='--',label='Nominal 90%')
axes[0].set(xlabel='Forecast lead (weeks)',ylabel='Observed 90% interval coverage (%)',ylim=(0,101),xticks=range(1,9))
axes[1].set(xlabel='Forecast lead (weeks)',ylabel='Held-out mean-forecast RMSE (ppm)',xticks=range(1,9))
for ax in axes:ax.grid(axis='y',alpha=.15)
axes[0].legend(frameon=False,loc='upper right',fontsize=8,ncol=2)
fig.suptitle('Mauna Loa: nominal 90% intervals are severely undercalibrated',fontweight='bold',fontsize=12)
for ext in ('pdf','png'):fig.savefig(out/('CO2_Heldout_Calibration_Failure.'+ext),dpi=220,bbox_inches='tight')
plt.close(fig)
real=load(root/'pass27/package/clinical_cvae_executed_results/test_predictions.csv')
fig,axes=plt.subplots(1,2,figsize=(10.5,4.25),layout='constrained')
for arm,color,label,marker in [('real_only',colors[0],'Real only','o'),
    ('real_bootstrap',colors[2],'Real + repeated real','D'),
    ('real_plus_cvae',colors[1],'Real + CVAE','o')]:
 aucs=[];balances=[]
 for seed in (11,23,37):
  q=[r for r in real if r['arm']==arm and int(r['seed'])==seed]
  assert len(q)==114
  y=np.array([int(r['true_label']) for r in q])
  p=np.array([float(r['probability_benign']) for r in q])
  fpr,tpr,_=roc_curve(y,p)
  if seed==11 or arm!='real_only':
   axes[0].plot(fpr,tpr,color=color,linewidth=2 if seed==11 else 1.2,
    alpha=.85 if seed==11 else .42,label=(label if seed==11 else None))
  # Scores from genuine held-out predictions, not summary-file literals
  from sklearn.metrics import roc_auc_score,balanced_accuracy_score
  auc=roc_auc_score(y,p);pred=(p>=.5).astype(int);balanced=balanced_accuracy_score(y,pred)
  aucs.append(auc);balances.append(balanced)
  axes[1].scatter(auc,balanced,color=color,marker=marker,s=77,
   edgecolor='white',linewidth=.7,zorder=3)
 axes[1].scatter(np.mean(aucs),np.mean(balances),marker='+',s=180,color=color,linewidth=2.4)
axes[0].plot([0,1],[0,1],ls='--',color='#BBC8D0',lw=1)
axes[0].set(xlim=(-.01,.4),ylim=(.6,1.01),xlabel='False positive rate',ylabel='True positive rate')
axes[0].legend(frameon=False,fontsize=8,loc='lower right')
axes[1].set(xlim=(.96,1.003),ylim=(.9,1.01),xlabel='Held-out ROC AUC',ylabel='Held-out balanced accuracy')
for ax in axes:ax.grid(alpha=.17)
fig.suptitle('Cytology: CVAE data augmentation degrades held-out discrimination',fontweight='bold',fontsize=12)
for ext in ('pdf','png'):fig.savefig(out/('Cytology_Heldout_Augmentation_Diagnostic.'+ext),dpi=220,bbox_inches='tight')
plt.close(fig)
(out/'co2_horizon_metrics.json').write_text(json.dumps(diagnostics,indent=2))
print('PASS: two scientific figures generated from genuine held-out experimental predictions')
