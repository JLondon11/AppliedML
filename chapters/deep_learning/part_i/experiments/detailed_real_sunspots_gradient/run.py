"""Observed-data PyTorch recurrent input-gradient experiment for detailed Chapter 2.

Original NGDC sunspot annual observations, 1700–2008, via statsmodels.
18 untrained initializations: RNN/LSTM/GRU × seeds 17/42/84 × 12/36-year
windows. Stores 24 actual-window input-gradient vectors per run.
No trained prediction or generalization claim.
"""
from pathlib import Path
import argparse, json, hashlib, random
import numpy as np
import torch
from torch import nn
from statsmodels.datasets import sunspots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class Cell(nn.Module):
    def __init__(self,kind):
        super().__init__()
        recurrent={
            "RNN":lambda:nn.RNN(1,16,batch_first=True,nonlinearity="tanh"),
            "LSTM":lambda:nn.LSTM(1,16,batch_first=True),
            "GRU":lambda:nn.GRU(1,16,batch_first=True)}[kind]
        self.core=recurrent()
        self.head=nn.Linear(16,1)
    def forward(self,x):
        states,_=self.core(x)
        return self.head(states[:,-1,:])

def run(out):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    raw=sunspots.load_pandas().data
    values=raw.SUNACTIVITY.to_numpy(dtype=np.float32)
    years=raw.YEAR.to_numpy(dtype=int)
    assert len(values)==309 and tuple(years[[0,-1]])==(1700,2008)
    mu=float(values[:190].mean())
    sd=float(values[:190].std())
    records=[]
    for window in [12,36]:
        X=np.stack([values[i-window:i] for i in range(window,len(values))])
        y=values[window:]
        target_years=years[window:]
        fit_idx=np.where(target_years<=1889)[0]
        assert len(fit_idx)==190-window
        chosen=fit_idx[np.linspace(0,len(fit_idx)-1,24).astype(int)]
        xnorm=((X-mu)/sd).astype(np.float32)
        ynorm=((y-mu)/sd).astype(np.float32)
        for seed in [17,42,84]:
            for kind in ["RNN","LSTM","GRU"]:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                model=Cell(kind)
                x=torch.tensor(xnorm[chosen,:,None],requires_grad=True)
                y_true=torch.tensor(ynorm[chosen,None])
                loss=nn.MSELoss()(model(x),y_true)
                loss.backward()
                grad=np.abs(x.grad.detach().numpy()[:,:,0])*len(chosen)
                per_lag=np.median(grad,axis=0)[::-1]
                records.append({
                    "window":window,"seed":seed,"model":kind,
                    "initial_squared_loss":float(loss.detach()),
                    "input_abs_gradient_by_training_window":grad[:,::-1].tolist(),
                    "median_absolute_gradient_lag_1_to_N":per_lag.tolist(),
                    "earliest_to_latest_ratio":float(per_lag[-1]/max(per_lag[0],1e-20))})
    evidence={
        "source":"NGDC annual sunspots via statsmodels.datasets.sunspots",
        "source_url":"https://www.statsmodels.org/stable/datasets/generated/sunspots.html",
        "source_sha256_float32":hashlib.sha256(values.tobytes()).hexdigest(),
        "original_observations":309,"original_date_span":[1700,2008],
        "train_end_year":1889,"train_only_normalization":[mu,sd],
        "real_training_windows_per_execution":24,
        "seeds":[17,42,84],"architectures":["RNN","LSTM","GRU"],
        "window_lengths":[12,36],"model_execution_count":18,
        "metric":"Median absolute squared-loss input gradient by lag on genuine original observed sunspot training windows at initialization",
        "limitations":"Input gradients at random initialization; not trained gradient-flow robustness or predictive accuracy",
        "records":records}
    (out/"original_sunspot_gradient_evidence.json").write_text(json.dumps(evidence,indent=2)+"\n")
    colors={"RNN":"#324C70","LSTM":"#BD7652","GRU":"#257A73"}
    fig,axes=plt.subplots(1,2,figsize=(11.6,4.5),layout="constrained")
    for ax,window in zip(axes,[12,36]):
        for kind in ("RNN","LSTM","GRU"):
            curves=np.array([r["median_absolute_gradient_lag_1_to_N"]
                for r in records if r["window"]==window and r["model"]==kind])
            curve=np.exp(np.log(np.maximum(curves,1e-18)).mean(axis=0))
            lag=np.arange(1,window+1)
            ax.plot(lag,curve,color=colors[kind],lw=2.2,label=kind)
            ax.fill_between(lag,curves.min(axis=0),curves.max(axis=0),
                color=colors[kind],alpha=.085)
        ax.set(title=f"({'a' if window==12 else 'b'}) {window}-year original-data windows",
            xlabel="Input lag (years; 1 is most recent)",
            ylabel="Median |input gradient|",yscale="log",xlim=(1,window))
        ax.grid(alpha=.16,which="both")
        ax.legend(frameon=False,ncol=3,fontsize=9)
    fig.suptitle("Initial recurrent-gradient sensitivity on genuine annual sunspot observations",
        fontsize=13.5,fontweight="bold",color="#243D58")
    for ext in ("png","pdf","svg"):
        fig.savefig(out/("Figure_2_Observed_Sunspots_Recurrent_Input_Gradient."+ext),
           dpi=250,bbox_inches="tight")
    plt.close(fig)
    summary={"executed_initializations":len(records),
       "original_observations":len(values),
       "median_36y_oldest_to_newest_ratio":{
            kind:float(np.median([r["earliest_to_latest_ratio"]
                for r in records if r["window"]==36 and r["model"]==kind]))
            for kind in ("RNN","LSTM","GRU")}}
    print(json.dumps(summary,indent=2))
    return evidence

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="outputs")
    run(p.parse_args().out)
