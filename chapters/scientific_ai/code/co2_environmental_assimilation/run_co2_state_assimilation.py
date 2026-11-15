#!/usr/bin/env python3
"""Real-data sequential state assimilation on weekly Mauna Loa CO2, statsmodels dataset.
Annual trend + harmonics fitted only on training data. A random-walk state corrects residual drift.
Prospective predictions use observations strictly before each prediction time.
"""
import csv, json, hashlib, platform
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
for s in ['results','figures']: (ROOT/s).mkdir(parents=True,exist_ok=True)
D=sm.datasets.co2.load_pandas().data.copy()
y=D['co2'].to_numpy(dtype=float)
dates=pd.DatetimeIndex(D.index)
t=np.arange(len(y),dtype=float)
assert len(y)==2284 and dates[0].year==1958 and dates[-1].year==2001
obs=np.isfinite(y)
train=(dates.year<=1993)
val=(dates.year>=1994)&(dates.year<=1997)
test=(dates.year>=1998)
assert train.sum()>1000 and val.sum()>100 and test.sum()>100
period=365.2425/7.0
X=np.column_stack([np.ones(len(t)),t/52.1775,
                   np.sin(2*np.pi*t/period),np.cos(2*np.pi*t/period),
                   np.sin(4*np.pi*t/period),np.cos(4*np.pi*t/period)])
coeff=np.linalg.lstsq(X[train&obs],y[train&obs],rcond=None)[0]
season_trend=X@coeff
train_residual=y[train&obs]-season_trend[train&obs]
r=max(float(np.var(train_residual,ddof=1)*0.45),0.02)
def sequential(q):
    mu=0.0; var=float(np.var(train_residual,ddof=1))
    forecasts=np.full(len(y),np.nan)
    sigmas=np.full(len(y),np.nan)
    posterior=np.full(len(y),np.nan)
    for j in range(len(y)):
        prior=mu; pv=var+q
        forecasts[j]=season_trend[j]+prior
        sigmas[j]=np.sqrt(pv+r)
        if obs[j]:
            gain=pv/(pv+r)
            mu=prior+gain*(y[j]-forecasts[j])
            var=(1-gain)*pv
        else:mu=prior; var=pv
        posterior[j]=season_trend[j]+mu
    return forecasts,sigmas,posterior
candidates=[0.001,0.005,0.02,0.05,0.1,0.25,0.5,1.0]
validation=[]
for q in candidates:
    pred,_,_=sequential(q)
    score=np.sqrt(np.mean((y[val&obs]-pred[val&obs])**2))
    validation.append({'q_ppm2_per_week':q,'validation_rmse_ppm':round(float(score),6)})
selected=min(validation,key=lambda z:z['validation_rmse_ppm'])['q_ppm2_per_week']
pred,sd,post=sequential(selected)
naive=np.full(len(y),np.nan)
for j in range(52,len(y)):
    for lag in [52,51,53,50,54]:
        if j-lag>=0 and np.isfinite(y[j-lag]):
            naive[j]=y[j-lag];break
mask=test & obs
metrics=[]
for name, forecast in [('Training-fitted trend + harmonics',season_trend),
                       ('Seasonal naive, year lag',naive),
                       ('Sequential Kalman residual assimilation',pred)]:
    m=mask&np.isfinite(forecast)
    metrics.append({'model':name,'test_observed_n':int(m.sum()),
                    'test_RMSE_ppm':round(float(np.sqrt(np.mean((y[m]-forecast[m])**2))),5),
                    'test_MAE_ppm':round(float(np.mean(np.abs(y[m]-forecast[m]))),5),
                    'test_bias_ppm':round(float(np.mean(forecast[m]-y[m])),5)})
coverage=np.mean((y[mask]>=pred[mask]-1.645*sd[mask])&(y[mask]<=pred[mask]+1.645*sd[mask]))
metrics[-1]['nominal_90pct_PI_coverage']=round(float(coverage),5)
metrics[-1]['nominal_90pct_PI_mean_width_ppm']=round(float(np.mean(2*1.645*sd[mask])),5)
with (ROOT/'results'/'co2_validation_grid.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(validation[0]));w.writeheader();w.writerows(validation)
with (ROOT/'results'/'co2_test_metrics.csv').open('w',newline='') as f:
 keys=list(dict.fromkeys(k for row in metrics for k in row))
 w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(metrics)
with (ROOT/'results'/'co2_test_predictions.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['date','observed_co2_ppm','harmonic_baseline_ppm','seasonal_naive_ppm','kalman_one_step_ppm','kalman_pi90_lower_ppm','kalman_pi90_upper_ppm','kalman_filtered_posterior_ppm'])
 for j in np.where(test)[0]:
    w.writerow([dates[j].date().isoformat(), *(round(float(x),6) if np.isfinite(x) else '' for x in [y[j],season_trend[j],naive[j],pred[j],pred[j]-1.645*sd[j],pred[j]+1.645*sd[j],post[j]])])
manifest={'dataset':'statsmodels Mauna Loa weekly atmospheric CO2; derived from historical Scripps/Keeling record',
 'source_url':'https://www.statsmodels.org/stable/datasets/generated/co2.html',
 'source_citation':'Keeling and Whorf (2004), as documented by statsmodels',
 'n_timestamps':int(len(y)),'n_observed':int(obs.sum()),'n_missing':int((~obs).sum()),
 'train_period':'1958-03 through 1993-12','validation_period':'1994-01 through 1997-12',
 'test_period':'1998-01 through 2001-12',
 'train_observed':int((train&obs).sum()), 'val_observed':int((val&obs).sum()),'test_observed':int(mask.sum()),
 'weekly_period_weeks':period,'fit_model':'OLS secular trend and first two annual Fourier harmonics on training only',
 'state_model':'random-walk residual + univariate Kalman filter, observes previous measurement only',
 'measurement_r_estimate_ppm2':r,'validation_selected_process_q_ppm2':selected,
 'metrics':metrics,'python':platform.python_version(),'numpy':np.__version__,'statsmodels':sm.__version__,
 'limitations':'One-station sequential nowcast-style one-step prediction, not a validated multi-physics digital twin; no external-station validation or causal interpretation. Assumed Gaussian predictive intervals are not calibrated.'}
(ROOT/'results'/'co2_execution_manifest.json').write_text(json.dumps(manifest,indent=2))
COL={'real':'#243746','kalman':'#167c80','naive':'#b27b3c','base':'#7b6b96'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,1,figsize=(11.3,7.1),layout='constrained',sharex=True)
region=test&(dates.year>=1999)
axs[0].plot(dates[region],y[region],color=COL['real'],lw=1.3,label='Measured weekly CO2',zorder=4)
axs[0].plot(dates[region],pred[region],color=COL['kalman'],lw=1.4,label='Kalman one-step prior prediction')
axs[0].plot(dates[region],naive[region],color=COL['naive'],lw=.95,alpha=.85,label='Year-lag naive')
axs[0].fill_between(dates[region],pred[region]-1.645*sd[region],pred[region]+1.645*sd[region],color=COL['kalman'],alpha=.16,label='Nominal 90% interval')
axs[0].set(ylabel='CO2 concentration (ppm)',title='(a) Real observations and prospective weekly one-step predictions')
axs[0].legend(loc='upper left',ncol=2,fontsize=8,frameon=False)
axs[1].axhline(0,color='#9098a0',lw=.9)
axs[1].plot(dates[region],pred[region]-y[region],color=COL['kalman'],lw=1.3,label='Kalman residual')
axs[1].plot(dates[region],naive[region]-y[region],color=COL['naive'],lw=1.0,label='Seasonal naive residual')
axs[1].set(ylabel='Forecast minus observed (ppm)',xlabel='Date',title='(b) Out-of-sample forecast errors')
axs[1].legend(frameon=False)
for ax in axs:ax.grid(alpha=.18)
for suffix in ['png','pdf']:fig.savefig(ROOT/'figures'/f'CO2_real_observation_assimilation.{suffix}',dpi=230,bbox_inches='tight')
plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(11.3,4.1),layout='constrained')
labels=['Trend + Fourier','Year-lag naive','Kalman residual']
vals=[x['test_RMSE_ppm'] for x in metrics]
xs=np.arange(3)
axs[0].plot(xs,vals,marker='o',markersize=7,lw=1.5,color=COL['kalman'])
for j,v in enumerate(vals):axs[0].annotate(f'{v:.2f}',(xs[j],v),xytext=(0,9),textcoords='offset points',ha='center')
axs[0].set(xticks=xs,xticklabels=labels,ylabel='Test RMSE (ppm)',title='(a) Real-data held-out error')
axs[0].grid(alpha=.17,axis='y')
axs[1].plot([x['q_ppm2_per_week'] for x in validation],[x['validation_rmse_ppm'] for x in validation],'-o',color=COL['naive'])
axs[1].axvline(selected,color=COL['kalman'],ls='--',label=f'Selected q={selected}')
axs[1].set(xscale='log',xlabel='Random-walk process variance q (ppm squared/week)',ylabel='Validation RMSE (ppm)',title='(b) Hyperparameter selection on validation period')
axs[1].grid(alpha=.17);axs[1].legend(frameon=False)
for suffix in ['png','pdf']:fig.savefig(ROOT/'figures'/f'CO2_real_validation_benchmarks.{suffix}',dpi=230,bbox_inches='tight')
plt.close(fig)
hashes={}
for f in sorted((ROOT/'results').glob('*.csv'))+sorted((ROOT/'figures').glob('*.png')):
 hashes[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
(ROOT/'results'/'sha256_evidence.json').write_text(json.dumps(hashes,indent=2))
print(json.dumps({'counts':{k:manifest[k] for k in ['n_timestamps','n_observed','n_missing','train_observed','val_observed','test_observed']},'selected_q':selected,'metrics':metrics},indent=2))
