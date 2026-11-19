"""Observed AirPassengers 1949–1960: LSTM one-step and truly recursive 24-month test.

Download dataset from https://github.com/jbrownlee/Datasets, original Git
blob 3751f1359045b83e41406b3630686fb2639a448c.
Run: python run.py --data airline-passengers.csv --output outputs
Requires numpy, torch, matplotlib. The original source contains 144 monthly observations.
"""
import argparse,csv,copy,hashlib,json,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED=1729;WINDOW=12;TRAIN_END=96;VAL_END=120
class Net(nn.Module):
 def __init__(self):
  super().__init__();self.rnn=nn.LSTM(1,16,batch_first=True);self.head=nn.Linear(16,1)
 def forward(self,x):return self.head(self.rnn(x)[0][:,-1,:])
def run(data_file,output_dir):
 out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
 with open(data_file) as f:rows=list(csv.DictReader(f))
 dates=[r["Month"] for r in rows]
 values=np.array([float(r["Passengers"]) for r in rows],dtype=np.float32)
 assert len(values)==144 and dates[0]=="1949-01" and dates[-1]=="1960-12"
 assert values.sum()==40363 and values[0]==112 and values[-1]==432
 torch.set_num_threads(2);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
 mean=float(values[:TRAIN_END].mean());std=float(values[:TRAIN_END].std())
 z=(values-mean)/std
 X=np.stack([z[t-WINDOW:t] for t in range(WINDOW,144)]).astype(np.float32)
 target=z[WINDOW:];times=np.arange(WINDOW,144)
 I=np.flatnonzero(times<TRAIN_END);V=np.flatnonzero((times>=TRAIN_END)&(times<VAL_END));J=np.flatnonzero(times>=VAL_END)
 assert tuple(map(len,(I,V,J)))==(84,24,24)
 x=torch.tensor(X[:,:,None]);y=torch.tensor(target[:,None])
 model=Net();opt=torch.optim.Adam(model.parameters(),lr=.007);mse=nn.MSELoss()
 best=float("inf");best_epoch=0;state=None;history=[]
 for epoch in range(1,451):
  model.train()
  order=I[torch.randperm(len(I),generator=torch.Generator().manual_seed(SEED*1000+epoch)).numpy()]
  for batch in np.array_split(order,int(np.ceil(len(order)/24))):
   opt.zero_grad();loss=mse(model(x[batch]),y[batch]);loss.backward()
   torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
  model.eval()
  with torch.no_grad():
   t_loss=float(mse(model(x[I]),y[I]))
   v_loss=float(mse(model(x[V]),y[V]))
  history.append({"epoch":epoch,"train_normalized_mse":t_loss,"validation_normalized_mse":v_loss})
  if v_loss<best-1e-6:
   best=v_loss;best_epoch=epoch;state=copy.deepcopy(model.state_dict())
  if epoch-best_epoch>=45:break
 model.load_state_dict(state);model.eval()
 with torch.no_grad():one=model(x[J]).squeeze(-1).numpy()*std+mean
 buffer=list(z[VAL_END-WINDOW:VAL_END].astype(float));recursive=[]
 for step in range(24):
  window=torch.tensor(np.array(buffer[-WINDOW:],dtype=np.float32)[None,:,None])
  with torch.no_grad():pred=float(model(window).item())
  buffer.append(pred);recursive.append(pred*std+mean)
 recursive=np.asarray(recursive);observed=values[VAL_END:]
 refs={
  "seasonal_one_step":values[VAL_END-12:144-12],
  "persistence_one_step":values[VAL_END-1:143],
  "seasonal_recursive_24_month":np.tile(values[108:120],2),
  "persistence_recursive_24_month":np.full(24,values[119])}
 preds={"LSTM_one_step_observed_history":one,"LSTM_recursive_24_month":recursive,**refs}
 def metrics(p):
  return {"rmse":float(np.sqrt(np.mean((observed-p)**2))),
          "mae":float(np.mean(np.abs(observed-p))),
          "bias":float(np.mean(p-observed))}
 scores={k:metrics(v) for k,v in preds.items()}
 result={
  "source":"AirPassengers original monthly counts 1949–1960",
  "upstream":"https://github.com/jbrownlee/Datasets/blob/master/airline-passengers.csv",
  "upstream_git_blob":"3751f1359045b83e41406b3630686fb2639a448c",
  "local_file_sha256":hashlib.sha256(Path(data_file).read_bytes()).hexdigest(),
  "dates":dates,"original_observations":values.astype(int).tolist(),
  "train_windows":84,"validation_windows":24,"test_windows":24,
  "training_months":96,"validation_months":24,"test_months":24,"window_size":12,
  "train_only_normalization":{"mean":mean,"std":std},
  "seed":SEED,"best_validation_epoch":best_epoch,"history":history,
  "test_target_months":dates[120:],"test_targets":observed.tolist(),
  "predictions":{k:v.tolist() for k,v in preds.items()},
  "scores":scores,
  "scope":"Historical offline single-series 24-month evaluation; recursive LSTM never observes 1959–1960 inputs; one seed"}
 (out/"measured_airpassengers_forecast.json").write_text(json.dumps(result,indent=2)+"\n")
 plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9.6,
  "axes.spines.top":False,"axes.spines.right":False})
 fig,axs=plt.subplots(2,2,figsize=(12,8.4),layout="constrained")
 navy="#233f59";teal="#2b8279";copper="#bd7650"
 axs[0,0].plot(range(144),values,c=navy,lw=1.8)
 for lo,hi,col,label in [(0,95,"#e8eef2","Training"),(96,119,"#f5e4d9","Validation"),(120,143,"#d9ece9","Test")]:
  axs[0,0].axvspan(lo,hi,color=col,label=label)
 axs[0,0].set(title="(a) Original observed monthly records",xlabel="Month (1949 = 0)",ylabel="Thousands of passengers")
 axs[0,0].legend(frameon=False,ncol=3,fontsize=8)
 for ax,forecast,reference,lead,title in [
    (axs[0,1],one,refs["seasonal_one_step"],"one","(b) One-step with observed input history"),
    (axs[1,0],recursive,refs["seasonal_recursive_24_month"],"recursive","(c) Fully recursive 24-month forecast")]:
  ax.plot(range(120,144),observed,c=navy,label="Observed",lw=2)
  ax.plot(range(120,144),forecast,c=teal,label=f"LSTM RMSE {metrics(forecast)['rmse']:.1f}",lw=1.8)
  ax.plot(range(120,144),reference,c=copper,ls="--",label=f"Seasonal RMSE {metrics(reference)['rmse']:.1f}",lw=1.4)
  ax.set(title=title,xlabel="Held-out month (1959–1960)",ylabel="Thousands of passengers")
  ax.legend(frameon=False,fontsize=8)
 axs[1,1].plot([v["epoch"] for v in history],[v["train_normalized_mse"] for v in history],c=navy,label="Train")
 axs[1,1].plot([v["epoch"] for v in history],[v["validation_normalized_mse"] for v in history],c=copper,label="Validation")
 axs[1,1].axvline(best_epoch,color=teal,ls="--",label=f"Selected {best_epoch}")
 axs[1,1].set(title="(d) Executed training and checkpoint selection",xlabel="Epoch",ylabel="Normalized MSE")
 axs[1,1].legend(frameon=False,fontsize=8)
 for ax in axs.flat:ax.grid(alpha=.13)
 fig.suptitle("Observed AirPassengers: one-step versus recursive LSTM test forecasts",
  color=navy,fontweight="bold",fontsize=14)
 for ext in ("png","pdf","svg"):
  fig.savefig(out/("Figure_2_AirPassengers_RealData_LSTM_Forecast."+ext),dpi=230,bbox_inches="tight")
 plt.close(fig)
 print(json.dumps(scores,indent=2))
if __name__=="__main__":
 parser=argparse.ArgumentParser()
 parser.add_argument("--data",required=True)
 parser.add_argument("--output",default="outputs")
 a=parser.parse_args()
 run(a.data,a.output)
