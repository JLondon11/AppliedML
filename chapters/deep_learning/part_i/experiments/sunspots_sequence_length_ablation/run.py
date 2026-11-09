"""Real-data annual-sunspot RNN / LSTM / GRU lookback-length ablation.
18 seeded training runs, train/validation/test chronologically partitioned.
"""
import copy,json,random,hashlib,platform
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import mean_absolute_error,mean_squared_error
from statsmodels.datasets import sunspots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
SEEDS=(17,42,84);LENGTHS=(12,36);ARCHS=("RNN","LSTM","GRU")
MAX_EPOCHS=80;PATIENCE=12;HIDDEN=16;LR=.006;BATCH=32
OUT=Path(__file__).resolve().parent/"outputs";OUT.mkdir(exist_ok=True,parents=True)
torch.set_num_threads(2)
df=sunspots.load_pandas().data
raw=df.SUNACTIVITY.to_numpy(dtype=np.float32)
years=df.YEAR.to_numpy(dtype=np.int32)
assert len(raw)==309 and years[0]==1700 and years[-1]==2008
source_hash=hashlib.sha256(raw.tobytes()).hexdigest()
train_observations=raw[years<=1889]
mu=float(np.mean(train_observations));sd=float(np.std(train_observations))
class Forecaster(nn.Module):
 def __init__(self,arch):
  super().__init__()
  cls={"RNN":nn.RNN,"LSTM":nn.LSTM,"GRU":nn.GRU}[arch]
  kwargs={"batch_first":True}
  if arch=="RNN":kwargs["nonlinearity"]="tanh"
  self.rec=cls(1,HIDDEN,**kwargs);self.head=nn.Linear(HIDDEN,1)
 def forward(self,x):
  h,_=self.rec(x);return self.head(h[:,-1,:])
def metrics(a,b):
 return {"MAE":float(mean_absolute_error(a,b)),"RMSE":float(np.sqrt(mean_squared_error(a,b)))}
runs=[];histories={};predictions={};test_targets={};test_years={};baselines={};partition={}
for length in LENGTHS:
 features=np.stack([raw[i-length:i] for i in range(length,len(raw))])
 targets=raw[length:];yr=years[length:]
 tr=np.where(yr<=1889)[0]
 va=np.where((yr>1889)&(yr<=1949))[0]
 te=np.where(yr>1949)[0]
 assert (len(tr),len(va),len(te))==(190-length,60,59)
 assert not (set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te))
 X=torch.tensor(((features-mu)/sd)[:,:,None],dtype=torch.float32)
 Y=torch.tensor(((targets-mu)/sd)[:,None],dtype=torch.float32)
 test_targets[str(length)]=targets[te].tolist();test_years[str(length)]=yr[te].tolist()
 baselines[str(length)]=metrics(targets[te],features[te,-1])
 partition[str(length)]={"train":len(tr),"validation":len(va),"test":len(te),
  "train_target_years":[int(yr[tr[0]]),int(yr[tr[-1]])],
  "validation_target_years":[int(yr[va[0]]),int(yr[va[-1]])],
  "test_target_years":[int(yr[te[0]]),int(yr[te[-1]])]}
 for arch in ARCHS:
  for seed in SEEDS:
   random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
   model=Forecaster(arch);optimizer=torch.optim.Adam(model.parameters(),lr=LR)
   lossfn=nn.MSELoss();best=float("inf");best_epoch=0;best_state=None;stale=0;history=[]
   for epoch in range(1,MAX_EPOCHS+1):
    model.train()
    order=tr[torch.randperm(len(tr),generator=torch.Generator().manual_seed(seed*10000+length*100+epoch)).numpy()]
    for batch in np.array_split(order,int(np.ceil(len(order)/BATCH))):
     optimizer.zero_grad(set_to_none=True)
     lossfn(model(X[batch]),Y[batch]).backward()
     nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
    model.eval()
    with torch.inference_mode():
     tl=float(lossfn(model(X[tr]),Y[tr]))
     vl=float(lossfn(model(X[va]),Y[va]))
    history.append({"epoch":epoch,"train_mse_normalized":tl,"validation_mse_normalized":vl})
    if vl < best-1e-6:
     best=vl;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict());stale=0
    else:stale+=1
    if stale>=PATIENCE:break
   model.load_state_dict(best_state);model.eval()
   with torch.inference_mode():
    p=model(X[te]).flatten().numpy()*sd+mu
    pv=model(X[va]).flatten().numpy()*sd+mu
   key=f"{arch}_{length}_{seed}"
   predictions[key]=p.tolist();histories[key]=history
   rec={"architecture":arch,"lookback":length,"seed":seed,"epochs_completed":len(history),
    "best_epoch":best_epoch,"best_val_mse_normalized":best,
    "validation_metrics":metrics(targets[va],pv),
    "test_metrics":metrics(targets[te],p),
    "parameters":sum(q.numel() for q in model.parameters())}
   runs.append(rec)
   print(key,rec["test_metrics"],flush=True)
summary={}
for length in LENGTHS:
 summary[str(length)]={}
 for arch in ARCHS:
  rows=[r for r in runs if r["lookback"]==length and r["architecture"]==arch]
  summary[str(length)][arch]={k:{"mean":float(np.mean([r["test_metrics"][k] for r in rows])),
   "sample_sd":float(np.std([r["test_metrics"][k] for r in rows],ddof=1))}
   for k in ("MAE","RMSE")}
full={"dataset":"NGDC annual sunspots distributed by statsmodels",
 "source_url":"https://www.statsmodels.org/stable/datasets/generated/sunspots.html",
 "observations":309,"source_sha256_float32":source_hash,
 "train_only_mean":mu,"train_only_sd":sd,"partition_counts_and_years":partition,
 "seeds":list(SEEDS),"architectures":list(ARCHS),"lookbacks":list(LENGTHS),
 "hidden_units":HIDDEN,"optimizer":"Adam","lr":LR,"max_epochs":MAX_EPOCHS,
 "early_stopping_patience":PATIENCE,
 "checkpoint_criterion":"lowest validation MSE; held-out test never used for checkpoint selection",
 "protocol":"One-step predictions using actual preceding observed years, not recursive multiyear forecasts",
 "runs":runs,"summary":summary,"baseline":baselines,"predictions":predictions,
 "learning_curves":histories,"test_years":test_years,"test_targets":test_targets,
 "python":platform.python_version(),"torch":torch.__version__,
 "limitation":"Three seeds; single chronological partition; model parameter counts differ"}
(OUT/"results.json").write_text(json.dumps(full,indent=2))
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
 "axes.spines.top":False,"axes.spines.right":False})
colors={"RNN":"#4a768e","LSTM":"#356a9a","GRU":"#ba7653"}
fig,ax=plt.subplots(figsize=(7.5,4.6))
for arch in ARCHS:
 means=[summary[str(k)][arch]["RMSE"]["mean"] for k in LENGTHS]
 stds=[summary[str(k)][arch]["RMSE"]["sample_sd"] for k in LENGTHS]
 ax.errorbar([12,36],means,yerr=stds,marker="o",lw=2,capsize=4,
  color=colors[arch],label=arch)
ax.axhline(baselines["12"]["RMSE"],ls=":",color="#728074",label="Persistence")
ax.set(xlabel="Input history (years)",ylabel="Held-out test RMSE (annual sunspot number)",xticks=[12,36])
ax.grid(alpha=.18);ax.legend(frameon=False);fig.tight_layout()
fig.savefig(OUT/"lookback_rmse_comparison.png",dpi=230);fig.savefig(OUT/"lookback_rmse_comparison.pdf");plt.close(fig)
for length in LENGTHS:
 fig,ax=plt.subplots(figsize=(10,4.5))
 ax.plot(test_years[str(length)],test_targets[str(length)],label="Observed sunspots",lw=2.4,color="#253c50")
 for arch in ARCHS:
  ax.plot(test_years[str(length)],predictions[f"{arch}_{length}_17"],label=arch,lw=1.4,color=colors[arch])
 ax.set(xlabel="Test target year",ylabel="Annual sunspot number")
 ax.grid(alpha=.18);ax.legend(frameon=False,ncol=4,loc="upper center");fig.tight_layout()
 fig.savefig(OUT/f"heldout_forecasts_{length}.png",dpi=230)
 fig.savefig(OUT/f"heldout_forecasts_{length}.pdf");plt.close(fig)
fig,ax=plt.subplots(figsize=(7.7,4.6))
for arch in ARCHS:
 for length,style in [(12,"-"),(36,"--")]:
  curve=histories[f"{arch}_{length}_17"]
  ax.plot([r["epoch"] for r in curve],[r["validation_mse_normalized"] for r in curve],
    color=colors[arch],linestyle=style,lw=1.7,label=f"{arch} {length}-year")
ax.set(xlabel="Epoch",ylabel="Validation MSE (training-normalized)",yscale="log")
ax.grid(alpha=.18);ax.legend(frameon=False,ncol=2,fontsize=8);fig.tight_layout()
fig.savefig(OUT/"validation_histories.png",dpi=230);fig.savefig(OUT/"validation_histories.pdf");plt.close(fig)
