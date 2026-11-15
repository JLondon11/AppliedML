#!/usr/bin/env python3
"""Chapter 11: real-observation sequential CO2 assimilation, source-controlled reproduction.
Dataset: public-domain Keeling/Whorf weekly Mauna Loa CO2 as bundled with statsmodels.
This is a one-step monitoring demonstration, NOT a full atmospheric digital twin.
"""
from pathlib import Path
import csv, json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
(OUT/"results").mkdir(exist_ok=True)
(OUT/"figures").mkdir(exist_ok=True)
d=sm.datasets.co2.load_pandas().data
dates=pd.DatetimeIndex(d.index)
y=d.co2.to_numpy(dtype=float)
assert len(y)==2284
good=np.isfinite(y)
t=np.arange(len(y),dtype=float)
period=365.2425/7
X=np.column_stack([np.ones(len(t)), t/52.1775,
  np.sin(2*np.pi*t/period), np.cos(2*np.pi*t/period),
  np.sin(4*np.pi*t/period), np.cos(4*np.pi*t/period)])
tr=(dates.year<=1993)
va=(dates.year>=1994)&(dates.year<=1997)
te=(dates.year>=1998)
coef=np.linalg.lstsq(X[tr&good],y[tr&good],rcond=None)[0]
base=X@coef
v=float(np.var(y[tr&good]-base[tr&good],ddof=1))
r=max(.45*v,.02)
def run(q):
    mu=0.; var=v
    pred=np.full(len(y),np.nan)
    sd=np.full(len(y),np.nan)
    for j in range(len(y)):
        pv=var+q
        pred[j]=base[j]+mu
        sd[j]=np.sqrt(pv+r)
        if good[j]:
            k=pv/(pv+r)
            mu+=k*(y[j]-pred[j])
            var=(1-k)*pv
        else:var=pv
    return pred,sd
grid=[.001,.005,.02,.05,.1,.25,.5,1.]
val=[]
for q in grid:
    p,_=run(q)
    val.append((q,float(np.sqrt(np.mean((p[va&good]-y[va&good])**2)))))
q=min(val,key=lambda pair:pair[1])[0]
pred,sd=run(q)
naive=np.full(len(y),np.nan)
for j in range(52,len(y)):
    for lag in (52,51,53,50,54):
        if j-lag>=0 and good[j-lag]:
            naive[j]=y[j-lag];break
test=te&good
metrics=[]
for name,values in [("Training trend + harmonics",base),
                    ("Seasonal naive",naive),
                    ("Kalman residual state",pred)]:
    mask=test&np.isfinite(values)
    metrics.append({"model":name,"test_n":int(mask.sum()),
      "rmse_ppm":round(float(np.sqrt(np.mean((y[mask]-values[mask])**2))),5),
      "mae_ppm":round(float(np.mean(np.abs(y[mask]-values[mask]))),5)})
with (OUT/"results"/"metrics.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=metrics[0]);w.writeheader();w.writerows(metrics)
with (OUT/"results"/"predictions.csv").open("w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["date","observed_ppm","frozen_baseline_ppm","year_lag_ppm",
                "kalman_prior_ppm","pi90_lower","pi90_upper"])
    for j in np.where(te)[0]:
        w.writerow([str(dates[j].date())]+[
          "" if not np.isfinite(z) else round(float(z),6)
          for z in [y[j],base[j],naive[j],pred[j],pred[j]-1.645*sd[j],pred[j]+1.645*sd[j]]])
meta={"dataset":"Mauna Loa atmospheric CO2 weekly historical measurements",
      "source":"https://www.statsmodels.org/stable/datasets/generated/co2.html",
      "train_n":int((tr&good).sum()),"validation_n":int((va&good).sum()),
      "test_n":int(test.sum()),"selected_q":q,
      "coverage_nominal_90pct":float(np.mean(
        (y[test]>=pred[test]-1.645*sd[test])&
        (y[test]<=pred[test]+1.645*sd[test]))),
      "warning":"Not a multi-physics twin; intervals over-conservative; one-station test."}
(OUT/"results"/"manifest.json").write_text(json.dumps(meta,indent=2))
fig,ax=plt.subplots(2,1,figsize=(10,6.5),sharex=True,layout="constrained")
m=te&(dates.year>=1999)
ax[0].plot(dates[m],y[m],color="#243746",lw=1.3,label="Real observations")
ax[0].plot(dates[m],pred[m],color="#167c80",lw=1.4,label="Kalman prior")
ax[0].plot(dates[m],naive[m],color="#b27b3c",lw=1.,label="Year-lag baseline")
ax[0].fill_between(dates[m],pred[m]-1.645*sd[m],pred[m]+1.645*sd[m],
                   color="#167c80",alpha=.16,label="Nominal 90% interval")
ax[1].axhline(0,color="#999999",lw=.9)
ax[1].plot(dates[m],pred[m]-y[m],color="#167c80",label="Kalman error")
ax[1].plot(dates[m],naive[m]-y[m],color="#b27b3c",label="Naive error")
ax[0].set(ylabel="CO2 (ppm)",title="(a) Sequential real-data state assimilation")
ax[1].set(xlabel="Date",ylabel="Forecast - measured (ppm)",
          title="(b) Held-out one-week prediction error")
for a in ax:a.grid(alpha=.16);a.legend(frameon=False)
fig.savefig(OUT/"figures"/"CO2_assimilation.png",dpi=220)
fig.savefig(OUT/"figures"/"CO2_assimilation.svg")
plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(10,3.8),layout="constrained")
x=np.arange(len(metrics))
ax[0].plot(x,[v["rmse_ppm"] for v in metrics],"-o",color="#167c80")
ax[0].set(xticks=x,xticklabels=["Trend","Year-lag","Kalman"],
          ylabel="Test RMSE (ppm)",title="(a) Held-out measured error")
ax[1].plot([a for a,b in val],[b for a,b in val],"-o",color="#b27b3c")
ax[1].axvline(q,color="#167c80",linestyle="--")
ax[1].set(xscale="log",xlabel="Process variance q (ppm2/week)",
          ylabel="Validation RMSE (ppm)",title="(b) Validation-only tuning")
for a in ax:a.grid(alpha=.16)
fig.savefig(OUT/"figures"/"CO2_validation.png",dpi=220)
fig.savefig(OUT/"figures"/"CO2_validation.svg")
print(json.dumps({"metrics":metrics,**meta},indent=2))
