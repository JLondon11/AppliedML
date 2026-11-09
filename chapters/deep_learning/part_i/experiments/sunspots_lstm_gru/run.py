"""Real annual sunspots (NGDC) one-step-ahead LSTM vs GRU experiment.
Run with Python, NumPy, PyTorch, matplotlib, scikit-learn, statsmodels.
Data = statsmodels.datasets.sunspots (309 records, 1700–2008).
"""
from pathlib import Path
import json, copy, random, hashlib, platform
import numpy as np
import torch
from torch import nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.datasets import sunspots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent/"outputs"; OUT.mkdir(exist_ok=True)
SEEDS=(17,42,84); LOOKBACK=12; HIDDEN=16
TRAIN_END=1889; VAL_END=1949
EPOCHS=90; PATIENCE=15; BATCH=32; LR=.006
torch.set_num_threads(2)
raw=sunspots.load_pandas().data
values=raw.SUNACTIVITY.to_numpy(dtype=np.float32)
years=raw.YEAR.to_numpy(dtype=np.int32)
assert len(values)==309 and years[0]==1700 and years[-1]==2008
sha=hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
X=np.stack([values[i-LOOKBACK:i] for i in range(LOOKBACK,len(values))])
y=values[LOOKBACK:]; yr=years[LOOKBACK:]
tr=yr<=TRAIN_END; va=(yr>TRAIN_END)&(yr<=VAL_END); te=yr>VAL_END
assert (tr.sum(),va.sum(),te.sum())==(178,60,59)
mu=float(values[:TRAIN_END-years[0]+1].mean())
sd=float(values[:TRAIN_END-years[0]+1].std())
TX=torch.tensor(((X-mu)/sd).astype(np.float32)[:,:,None])
TY=torch.tensor(((y-mu)/sd).astype(np.float32)[:,None])
I=np.where(tr)[0]; V=np.where(va)[0]; J=np.where(te)[0]
class Forecast(nn.Module):
 def __init__(self,kind):
  super().__init__()
  self.rnn=(nn.LSTM(1,HIDDEN,batch_first=True) if kind=="LSTM" else nn.GRU(1,HIDDEN,batch_first=True))
  self.head=nn.Linear(HIDDEN,1)
 def forward(self,x):
  h,_=self.rnn(x); return self.head(h[:,-1])
def scores(actual,pred):
 return {"mae":float(mean_absolute_error(actual,pred)),"rmse":float(np.sqrt(mean_squared_error(actual,pred)))}
results=[]; histories={}; predictions={}
for seed in SEEDS:
 for kind in ("LSTM","GRU"):
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
  m=Forecast(kind); opt=torch.optim.Adam(m.parameters(),lr=LR)
  mse=nn.MSELoss(); best=float("inf"); best_state=None; best_epoch=0;stale=0; curve=[]
  for epoch in range(1,EPOCHS+1):
   m.train()
   generator=torch.Generator().manual_seed(seed*1000+epoch)
   order=I[torch.randperm(len(I),generator=generator).numpy()]
   for b in np.array_split(order,int(np.ceil(len(order)/BATCH))):
    opt.zero_grad(set_to_none=True)
    mse(m(TX[b]),TY[b]).backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(),max_norm=1.0)
    opt.step()
   m.eval()
   with torch.no_grad():
    tl=float(mse(m(TX[I]),TY[I]))
    vl=float(mse(m(TX[V]),TY[V]))
   curve.append({"epoch":epoch,"train_mse_normalized":tl,"validation_mse_normalized":vl})
   if vl<best-1e-6:
    best=vl; best_epoch=epoch; best_state=copy.deepcopy(m.state_dict());stale=0
   else:stale+=1
   if stale>=PATIENCE:break
  m.load_state_dict(best_state);m.eval()
  with torch.no_grad():
   p=m(TX[J]).squeeze(-1).numpy()*sd+mu
   vp=m(TX[V]).squeeze(-1).numpy()*sd+mu
  record={"architecture":kind,"seed":seed,"epochs_run":len(curve),"best_validation_epoch":best_epoch,
   "best_normalized_validation_mse":best,"validation":scores(y[va],vp),"test":scores(y[te],p),
   "parameter_count":sum(q.numel() for q in m.parameters())}
  results.append(record);histories[f"{kind}_{seed}"]=curve;predictions[f"{kind}_{seed}"]=p.tolist()
  print(kind,seed,record["test"],flush=True)
baseline=scores(y[te],X[J,-1])
aggregates={}
for kind in ("LSTM","GRU"):
 q=[r for r in results if r["architecture"]==kind]
 aggregates[kind]={k:{"mean":float(np.mean([z["test"][k] for z in q])),
  "sample_sd":float(np.std([z["test"][k] for z in q],ddof=1))} for k in ("mae","rmse")}
out={"dataset":"NGDC annual sunspot record via statsmodels.datasets.sunspots",
 "source_url":"https://www.statsmodels.org/stable/datasets/generated/sunspots.html",
 "observations_total":309,"year_range":[1700,2008],"target_windows_total":len(y),"lookback":LOOKBACK,
 "training_windows":int(tr.sum()),"validation_windows":int(va.sum()),"testing_windows":int(te.sum()),
 "training_target_years":[1712,1889],"validation_target_years":[1890,1949],"testing_target_years":[1950,2008],
 "normalization_train_mean":mu,"normalization_train_sd":sd,"source_values_sha256_float32":sha,
 "seeds":list(SEEDS),"model_selection":"Minimum validation MSE, test labels never used for selection",
 "results":results,"aggregates":aggregates,"persistence_baseline_test":baseline,
 "test_years":yr[te].tolist(),"ground_truth_test":y[te].tolist(),"forecasts":predictions,
 "training_histories":histories,"python_version":platform.python_version(),"torch_version":torch.__version__,
 "limitation":"12 observed previous annual values predict next observed year; not recursive multiyear rollout."}
(OUT/"results.json").write_text(json.dumps(out,indent=2))
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
fig,ax=plt.subplots(figsize=(10.2,4.5))
ax.plot(yr[te],y[te],lw=2.3,label="Observed annual sunspots",c="#233c55")
ax.plot(yr[te],predictions["LSTM_17"],lw=1.5,label="LSTM seed 17",c="#4373a8")
ax.plot(yr[te],predictions["GRU_17"],lw=1.5,label="GRU seed 17",c="#c37c53")
ax.set(xlabel="Target year (held-out period)",ylabel="Annual sunspot number")
ax.legend(frameon=False,ncol=3,loc="upper center");ax.grid(alpha=.17)
fig.tight_layout();fig.savefig(OUT/"sunspots_test_forecasts.png",dpi=220);fig.savefig(OUT/"sunspots_test_forecasts.pdf");plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4.4))
xx=np.arange(3)
for kind,color,offset in [("LSTM","#4373a8",-.11),("GRU","#c37c53",.11)]:
 v=[next(z for z in results if z["architecture"]==kind and z["seed"]==seed)["test"]["rmse"] for seed in SEEDS]
 ax.plot(xx+offset,v,marker="o",markersize=7,lw=1.8,c=color,label=kind)
ax.axhline(baseline["rmse"],color="#576f64",ls=":",label="Persistence baseline")
ax.set(xlabel="Initialization seed",ylabel="Test RMSE (annual sunspot number)",xticks=xx,xticklabels=[str(s) for s in SEEDS])
ax.grid(alpha=.17);ax.legend(frameon=False)
fig.tight_layout();fig.savefig(OUT/"sunspots_rmse_by_seed.png",dpi=220);fig.savefig(OUT/"sunspots_rmse_by_seed.pdf");plt.close(fig)
fig,ax=plt.subplots(figsize=(7.7,4.4))
for seed in SEEDS:
 for kind,c,ls in [("LSTM","#4373a8","-"),("GRU","#c37c53","--")]:
  hh=histories[f"{kind}_{seed}"]
  ax.plot([z["epoch"] for z in hh],[z["validation_mse_normalized"] for z in hh],lw=1.4,ls=ls,
   color=c,alpha=.5 if seed!=17 else .95,label=f"{kind} {seed}")
ax.set(xlabel="Epoch",ylabel="Validation MSE (train-normalized sunspots)",yscale="log")
ax.legend(frameon=False,ncol=2,fontsize=8);ax.grid(alpha=.17)
fig.tight_layout();fig.savefig(OUT/"sunspots_validation_histories.png",dpi=220)
fig.savefig(OUT/"sunspots_validation_histories.pdf");plt.close(fig)
print("AGGREGATE",aggregates,"BASELINE",baseline)
