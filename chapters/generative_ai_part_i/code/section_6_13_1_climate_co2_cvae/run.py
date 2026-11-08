"""Conditional VAE for real Mauna Loa weekly CO2 observations.

Predictive mean over 128 latent prior samples for eight weeks following 52
observed weeks. Strict chronological windows; observational data, not Earth
system model. Run: python run.py --epochs 65
"""
import csv,json,hashlib,random,argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import Ridge
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEEDS=(17,23,41)
H=52;F=8
TRAIN_END='1990-12-31';VAL_END='1995-12-31'

class CVAE(nn.Module):
    def __init__(self,hidden=64,zdim=8):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(H+F,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU())
        self.mu=nn.Linear(hidden,zdim)
        self.logvar=nn.Linear(hidden,zdim)
        self.dec=nn.Sequential(nn.Linear(H+zdim,hidden),nn.ReLU(),nn.Linear(hidden,F))
    def forward(self,x,y):
        features=self.enc(torch.cat((x,y),axis=-1))
        mu=self.mu(features);logvar=self.logvar(features).clamp(-8,8)
        z=mu+torch.exp(.5*logvar)*torch.randn_like(mu)
        return self.dec(torch.cat((x,z),axis=-1)),mu,logvar
    def generate(self,x,count=100,seed=1):
        rng=torch.Generator().manual_seed(seed)
        z=torch.randn((count,len(x),self.mu.out_features),generator=rng)
        xx=x[None].expand(count,-1,-1)
        with torch.no_grad():
            return self.dec(torch.cat((xx,z),axis=-1)).numpy()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=65)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'results')
    args=parser.parse_args();out=args.output;fig=out/'figures';fig.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    raw=sm.datasets.co2.load_pandas().data['co2'].asfreq('W-SAT')
    missing=int(raw.isna().sum())
    vals=raw.to_numpy(dtype=np.float32);dates=raw.index
    inputs=[];targets=[];starts=[]
    for i in range(H,len(vals)-F+1):
        hist=vals[i-H:i];future=vals[i:i+F]
        if np.isnan(hist).any() or np.isnan(future).any():continue
        inputs.append(hist);targets.append(future);starts.append(dates[i])
    X=np.stack(inputs);Y=np.stack(targets);starts=pd.DatetimeIndex(starts)
    train=np.flatnonzero(starts+pd.Timedelta(weeks=F-1)<=pd.Timestamp(TRAIN_END))
    val=np.flatnonzero((starts>TRAIN_END)&(starts+pd.Timedelta(weeks=F-1)<=pd.Timestamp(VAL_END)))
    test=np.flatnonzero(starts>VAL_END)
    assert len(train)>500 and len(val)>100 and len(test)>150
    Xn=(X-X[:,-1:])/10
    Yn=(Y-X[:,-1:])/10
    ridge=Ridge(alpha=100).fit(Xn[train],Yn[train])
    base_persist=np.repeat(X[:,-1:],F,axis=1)
    drift=X[:,-8:].mean(axis=1)-X[:,-52:-44].mean(axis=1)
    base_season=X[:,-52:-52+F]+drift[:,None]
    base_ridge=ridge.predict(Xn)*10+X[:,-1:]
    rows=[];sequences={};trace=[];coverages=[]
    for seed in SEEDS:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        net=CVAE();optimizer=torch.optim.Adam(net.parameters(),lr=0.001)
        xt=torch.from_numpy(Xn[train]);yt=torch.from_numpy(Yn[train])
        vv=torch.from_numpy(Xn[val]);vy=torch.from_numpy(Yn[val])
        best=float('inf');state=None;chosen=-1
        for epoch in range(args.epochs):
            net.train();order=torch.randperm(len(xt));total=[]
            for batch in order.split(64):
                pred,mu,lv=net(xt[batch],yt[batch])
                rec=((pred-yt[batch])**2).mean()
                kl=(-.5*(1+lv-mu**2-lv.exp()).sum(-1)).mean()
                loss=rec+.002*kl
                optimizer.zero_grad();loss.backward();optimizer.step()
                total.append(float(loss.detach()))
            net.eval()
            with torch.no_grad():
                pred=net.generate(vv,count=32,seed=10000+seed+epoch).mean(0)
                val_mse=float(np.mean((pred-vy.numpy())**2))
            trace.append((seed,epoch+1,float(np.mean(total)),val_mse))
            if val_mse<best:
                best=val_mse;chosen=epoch+1
                state={k:v.clone() for k,v in net.state_dict().items()}
        net.load_state_dict(state);net.eval()
        testx=torch.from_numpy(Xn[test])
        draws=net.generate(testx,count=128,seed=20000+seed)
        forecast=draws.mean(axis=0)*10+X[test,-1:]
        qlo=np.quantile(draws,.05,axis=0)*10+X[test,-1:]
        qhi=np.quantile(draws,.95,axis=0)*10+X[test,-1:]
        sequences[seed]=(forecast,qlo,qhi)
        torch.save({'weights':state,'seed':seed,'selected_epoch':chosen,'horizon':F,
                    'history_weeks':H,'dataset':'statsmodels Mauna Loa CO2'},
                   out/f'cvae_seed{seed}.pt')
        for method,pred in [('CVAE_prior_mean',forecast),('ridge',base_ridge[test]),
                            ('persistence',base_persist[test]),('seasonal_drift',base_season[test])]:
            rows.append({'seed':seed,'model':method,'n_test_windows':len(test),
                         'RMSE_ppm':float(np.sqrt(np.mean((pred-Y[test])**2))),
                         'MAE_ppm':float(np.abs(pred-Y[test]).mean()),
                         'selected_epoch':chosen if method=='CVAE_prior_mean' else ''})
        coverage=float(((Y[test]>=qlo)&(Y[test]<=qhi)).mean())
        coverages.append({'seed':seed,'nominal_coverage':0.90,'empirical_coverage':coverage,
                          'n_test_windows':len(test)})
    with (out/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (out/'interval_coverage.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(coverages[0]));w.writeheader();w.writerows(coverages)
    with (out/'training.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['seed','epoch','train_objective','validation_prior_MSE'])
        w.writerows(trace)
    with (out/'test_predictions.csv').open('w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['seed','first_forecast_week','horizon_week','true_co2_ppm',
                    'cvae_mean_ppm','cvae_q05_ppm','cvae_q95_ppm','ridge_ppm',
                    'persistence_ppm','seasonal_drift_ppm'])
        for seed in SEEDS:
            pred,lo,hi=sequences[seed]
            for j,index in enumerate(test):
                for k in range(F):
                    w.writerow([seed,str(starts[index].date()),k+1,float(Y[index,k]),
                        float(pred[j,k]),float(lo[j,k]),float(hi[j,k]),
                        float(base_ridge[index,k]),float(base_persist[index,k]),
                        float(base_season[index,k])])
    f1,ax=plt.subplots(figsize=(8,4.5))
    names=['CVAE_prior_mean','ridge','persistence','seasonal_drift']
    for i,n in enumerate(names):
        values=[r['RMSE_ppm'] for r in rows if r['model']==n]
        ax.scatter(i+np.array([-.07,0,.07]),values,s=30)
        ax.hlines(np.mean(values),i-.13,i+.13,linewidth=2)
    ax.set(xticks=np.arange(len(names)),
           xticklabels=['Conditional VAE','Ridge','Persistence','Seasonal drift'],
           ylabel='Held-out 8-week forecast RMSE (ppm)')
    ax.grid(axis='y',alpha=.2);f1.tight_layout()
    f1.savefig(fig/'climate_forecast_benchmark.png',dpi=200);plt.close(f1)
    f2,ax=plt.subplots(figsize=(8.2,4.2))
    seed=SEEDS[0];p,l,h=sequences[seed]
    chosen=np.arange(0,len(test),max(1,len(test)//12));x=starts[test[chosen]]
    ax.plot(x,Y[test[chosen],-1],'-o',label='Observed 8-week endpoint',markersize=3)
    ax.plot(x,p[chosen,-1],'-o',label='CVAE predictive mean',markersize=3)
    ax.fill_between(x,l[chosen,-1],h[chosen,-1],alpha=.2,label='Nominal central 90%')
    ax.set(ylabel='Atmospheric CO2 (ppm)',xlabel='Forecast start date')
    ax.legend(frameon=False,fontsize=8);f2.autofmt_xdate();f2.tight_layout()
    f2.savefig(fig/'climate_heldout_predictions.png',dpi=200);plt.close(f2)
    manifest={'source':'statsmodels.datasets.co2 Mauna Loa weekly atmospheric CO2 1958-2001',
      'raw_sha256':hashlib.sha256(np.nan_to_num(vals,nan=-999).tobytes()).hexdigest(),
      'n_observations':len(vals),'n_missing_original':missing,
      'n_complete_windows':len(X),'train_windows':len(train),
      'validation_windows':len(val),'test_windows':len(test),
      'split_rule':'all eight forecast targets lie inside assigned chronological interval',
      'chronological_train_end':TRAIN_END,'chronological_validation_end':VAL_END,
      'history_weeks':H,'forecast_horizon_weeks':F,'seeds':list(SEEDS),'epochs':args.epochs,
      'caveats':['overlapping windows dependent, not independent replicates',
                 'single observational CO2 series is not Earth-system simulation',
                 'prior stochastic samples lack calibrated uncertainty',
                 'windows with missing observations excluded',
                 'ridge trained only on training windows']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'manifest':manifest,'metrics':rows},indent=2))
if __name__=='__main__':main()
