"""Real-data recurrent input-gradient attenuation on NGDC annual sunspots.
Uses untrained initialized RNN, LSTM and GRU; reports gradient sensitivity, NOT accuracy.
"""
import json, random, hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from statsmodels.datasets import sunspots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent/"outputs"; OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
df=sunspots.load_pandas().data
values=df.SUNACTIVITY.to_numpy(dtype=np.float32)
years=df.YEAR.to_numpy(dtype=np.int32)
assert len(values)==309 and years[0]==1700 and years[-1]==2008
mu=float(values[:190].mean()); sd=float(values[:190].std())
SEEDS=(17,42,84); LENGTHS=(12,36); KINDS=("RNN","LSTM","GRU")
class Model(nn.Module):
 def __init__(self,kind):
  super().__init__()
  self.recurrent=(nn.RNN(1,16,batch_first=True,nonlinearity="tanh") if kind=="RNN" else
                  nn.LSTM(1,16,batch_first=True) if kind=="LSTM" else
                  nn.GRU(1,16,batch_first=True))
  self.head=nn.Linear(16,1)
 def forward(self,x):
  outputs,_=self.recurrent(x)
  return self.head(outputs[:,-1,:])
results=[]
for length in LENGTHS:
 X=np.stack([values[i-length:i] for i in range(length,len(values))])
 targets=values[length:]; targets_years=years[length:]
 train=np.where(targets_years<=1889)[0]
 val=np.where((targets_years>1889)&(targets_years<=1949))[0]
 test=np.where(targets_years>1949)[0]
 assert (len(train),len(val),len(test))==(190-length,60,59)
 Xn=((X-mu)/sd).astype(np.float32)
 yn=((targets-mu)/sd).astype(np.float32)
 selected=train[np.linspace(0,len(train)-1,24).astype(int)]
 for seed in SEEDS:
  for kind in KINDS:
   random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
   model=Model(kind)
   x=torch.tensor(Xn[selected,:,None],requires_grad=True)
   target=torch.tensor(yn[selected,None])
   loss=nn.MSELoss()(model(x),target)
   loss.backward()
   absolute_gradient=np.abs(x.grad.detach().numpy()[:,:,0])*len(selected)
   by_lag=np.median(absolute_gradient,axis=0)[::-1]
   result={"kind":kind,"seed":seed,"window_length":length,"samples_used_for_gradient":24,
     "train_windows":len(train),"validation_windows":len(val),"test_windows":len(test),
     "initial_batch_loss":float(loss.detach()),
     "median_abs_input_gradient_by_lag_1_to_length":by_lag.tolist(),
     "recent_lag_gradient":float(by_lag[0]),"earliest_lag_gradient":float(by_lag[-1]),
     "earliest_to_recent_ratio":float(by_lag[-1]/max(by_lag[0],1e-20))}
   results.append(result)
report={"dataset":"NGDC annual sunspot record via statsmodels.datasets.sunspots",
 "data_source":"https://www.statsmodels.org/stable/datasets/generated/sunspots.html",
 "observations_total":309,"years":[1700,2008],
 "train_observations_for_normalization":190,
 "training_target_end_year":1889,"validation_target_years":[1890,1949],
 "testing_target_years":[1950,2008],"seeds":list(SEEDS),
 "lengths":list(LENGTHS),"kinds":list(KINDS),
 "model":"single untrained recurrent layer with 16 units and linear regression head",
 "gradient_quantity":"Median absolute derivative of squared loss with respect to input at each lag; batch mean corrected by sample count; 24 genuine training windows",
 "limitations":"Initialization-only gradient sensitivity, no predictive accuracy, one dataset and three seeds",
 "observations_float32_sha256":hashlib.sha256(values.tobytes()).hexdigest(),
 "results":results}
(OUT/"results.json").write_text(json.dumps(report,indent=2))
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
 "axes.spines.top":False,"axes.spines.right":False})
colors={"RNN":"#315e8a","LSTM":"#c4794a","GRU":"#407e75"}
for length in LENGTHS:
 fig,ax=plt.subplots(figsize=(8.1,4.7))
 for kind in KINDS:
  gradients=np.array([v["median_abs_input_gradient_by_lag_1_to_length"]
   for v in results if v["kind"]==kind and v["window_length"]==length])
  geom=np.exp(np.log(np.maximum(gradients,1e-16)).mean(axis=0))
  ax.semilogy(range(1,length+1),geom,label=kind,c=colors[kind],lw=2)
 ax.set(xlabel="Lag (1 = most recent annual observation)",
        ylabel="Median absolute input gradient, geometric mean across seeds",
        title=f"Measured input gradients on real sunspot windows ({length} years)")
 ax.grid(alpha=.2);ax.legend(frameon=False)
 fig.tight_layout()
 fig.savefig(OUT/f"recurrent_input_gradient_{length}.png",dpi=230)
 fig.savefig(OUT/f"recurrent_input_gradient_{length}.pdf");plt.close(fig)
fig,ax=plt.subplots(figsize=(8.1,4.7))
for kind in KINDS:
 vals=[v["earliest_to_recent_ratio"] for v in results
       if v["kind"]==kind and v["window_length"]==36]
 ax.plot([1,2,3],vals,"-o",color=colors[kind],lw=2,label=kind)
ax.set(xlabel="Seed index (17, 42, 84)",
 ylabel="Oldest-lag / newest-lag gradient ratio",
 title="Input-gradient decay on real 36-year sequences",xticks=[1,2,3],yscale="log")
ax.legend(frameon=False);ax.grid(alpha=.2)
fig.tight_layout()
fig.savefig(OUT/"gradient_decay_by_seed.png",dpi=230)
fig.savefig(OUT/"gradient_decay_by_seed.pdf");plt.close(fig)
