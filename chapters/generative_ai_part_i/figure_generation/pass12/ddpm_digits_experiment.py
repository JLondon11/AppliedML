"""Measured DDPM on real sklearn digits (8x8). Unconditional samples and independent test metrics.

Run: python ddpm_digits_experiment.py --out ddpm_results --epochs 45
The model is a small teaching baseline, not a DDIM implementation or SOTA generator.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.nn import functional as F

T = 100


class NoisePredictor(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(66,hidden),nn.SiLU(),
                               nn.Linear(hidden,hidden),nn.SiLU(),
                               nn.Linear(hidden,64))

    def forward(self,x,t):
        phase=t.float().reshape(-1,1)/(T-1)
        condition=torch.cat([torch.sin(phase*2*np.pi),torch.cos(phase*2*np.pi)],dim=1)
        return self.net(torch.cat([x,condition],dim=1))


def schedule():
    beta=torch.linspace(0.0005,0.095,T)
    alpha=1-beta
    abar=torch.cumprod(alpha,dim=0)
    assert (abar[1:] < abar[:-1]).all()
    assert 0 < abar[-1].item() < .02
    return beta,alpha,abar


def q_sample(x0,t,noise,abar):
    a=abar[t].reshape(-1,1)
    return torch.sqrt(a)*x0+torch.sqrt(1-a)*noise


@torch.no_grad()
def evaluate(model,x,abar,seed=101,batch_size=256):
    model.eval()
    gen=torch.Generator().manual_seed(seed)
    aggregate={str(t):[0.,0.,0.] for t in [9,24,49,74,99]}
    for i in range(0,len(x),batch_size):
        xb=x[i:i+batch_size]
        for t in [9,24,49,74,99]:
            tt=torch.full((len(xb),),t,dtype=torch.long)
            eps=torch.randn(xb.shape,generator=gen)
            pred=model(q_sample(xb,tt,eps,abar),tt)
            a=aggregate[str(t)]
            a[0]+=F.mse_loss(pred,eps,reduction='sum').item()
            a[1]+=eps.square().sum().item()
            a[2]+=eps.numel()
    return {k:{'predicted_noise_mse':v[0]/v[2],
               'zero_predictor_mse':v[1]/v[2]} for k,v in aggregate.items()}


@torch.no_grad()
def sample_ddpm(model,count,beta,alpha,abar,seed):
    model.eval()
    gen=torch.Generator().manual_seed(seed)
    x=torch.randn((count,64),generator=gen)
    for i in range(T-1,-1,-1):
        times=torch.full((count,),i,dtype=torch.long)
        eps=model(x,times)
        mean=(x-beta[i]/torch.sqrt(1-abar[i])*eps)/torch.sqrt(alpha[i])
        if i>0:
            posterior_var=beta[i]*(1-abar[i-1])/(1-abar[i])
            x=mean+torch.sqrt(posterior_var)*torch.randn(x.shape,generator=gen)
        else:
            x=mean
    return x


def run(out,epochs=45,seed=17):
    if epochs < 1: raise ValueError('epochs must be positive')
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    digits=load_digits()
    real=digits.data.astype('float32')/8.0-1.0
    labels=digits.target
    all_idx=np.arange(len(real))
    trva_idx,test_idx=train_test_split(all_idx,test_size=270,stratify=labels,random_state=seed)
    tr_idx,val_idx=train_test_split(trva_idx,test_size=270,stratify=labels[trva_idx],random_state=seed)
    assert [len(v) for v in (tr_idx,val_idx,test_idx)]==[1257,270,270]
    assert len(set(tr_idx)|set(val_idx)|set(test_idx))==len(real)
    assert not(set(tr_idx)&set(val_idx) or set(tr_idx)&set(test_idx) or set(val_idx)&set(test_idx))
    xtr=torch.from_numpy(real[tr_idx])
    xval=torch.from_numpy(real[val_idx])
    xtest=torch.from_numpy(real[test_idx])
    beta,alpha,abar=schedule()
    model=NoisePredictor()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    rng=torch.Generator().manual_seed(seed+200)
    history=[]
    best=float('inf')
    best_epoch=0
    checkpoint=out/'best_ddpm.pt'
    for epoch in range(1,epochs+1):
        model.train()
        order=torch.randperm(len(xtr),generator=rng)
        total=0.
        for subset in order.split(128):
            x=xtr[subset]
            t=torch.randint(0,T,(len(x),),generator=rng)
            eps=torch.randn(x.shape,generator=rng)
            optimizer.zero_grad()
            loss=F.mse_loss(model(q_sample(x,t,eps,abar),t),eps)
            loss.backward()
            optimizer.step()
            total+=loss.item()*len(x)
        val=evaluate(model,xval,abar,seed=seed+900)
        val_score=float(np.mean([r['predicted_noise_mse'] for r in val.values()]))
        history.append({'epoch':epoch,'train_eps_mse':total/len(xtr),
                        'validation_mean_eps_mse':val_score})
        if val_score<best:
            best=val_score
            best_epoch=epoch
            torch.save({'model':model.state_dict(),'seed':seed,'epoch':epoch,'steps':T},checkpoint)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'])
    test_scores=evaluate(model,xtest,abar,seed=seed+1100)
    val_scores=evaluate(model,xval,abar,seed=seed+900)
    samples=sample_ddpm(model,16,beta,alpha,abar,seed+1300).numpy()
    measured=np.array([v['predicted_noise_mse'] for v in test_scores.values()])
    baseline=np.array([v['zero_predictor_mse'] for v in test_scores.values()])
    assert np.isfinite(samples).all() and np.isfinite(measured).all()
    raw_range=[float(samples.min()),float(samples.max())]
    shown=np.clip((samples+1)/2,0,1) # display clipping only
    fig,axes=plt.subplots(4,4,figsize=(5.3,5.3))
    for ax,pix in zip(axes.flat,shown):
        ax.imshow(pix.reshape(8,8),cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        ax.axis('off')
    fig.suptitle('Unconditional DDPM samples: trained on real digits',fontsize=11)
    fig.tight_layout();fig.savefig(out/'ddpm_generated_samples.png',dpi=190);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(8,4))
    for i,ax in enumerate(axes.flat):
        ax.imshow((real[test_idx[i]].reshape(8,8)+1)/2,cmap='gray',
                  vmin=0,vmax=1,interpolation='nearest')
        ax.axis('off')
    fig.suptitle('Independent held-out real digit images',fontsize=11)
    fig.tight_layout();fig.savefig(out/'real_test_digits.png',dpi=190);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(9,3.4))
    axs[0].plot([h['epoch'] for h in history],[h['train_eps_mse'] for h in history],label='train')
    axs[0].plot([h['epoch'] for h in history],[h['validation_mean_eps_mse'] for h in history],label='validation')
    axs[0].axvline(best_epoch,color='gray',linestyle='--',label='selected checkpoint')
    axs[0].set(xlabel='Epoch',ylabel='Noise-prediction MSE')
    axs[0].legend(fontsize=7)
    levels=np.array([10,25,50,75,100])
    axs[1].plot(levels,measured,marker='o',label='trained DDPM epsilon predictor')
    axs[1].plot(levels,baseline,marker='s',label='zero predictor baseline')
    axs[1].set(xlabel='Forward noise timestep',ylabel='Independent test epsilon MSE')
    axs[1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'ddpm_test_metrics.png',dpi=190);plt.close(fig)
    with (out/'training_history.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
    with (out/'test_metrics.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['timestep','model_epsilon_mse','zero_predictor_epsilon_mse'])
        for t,v in test_scores.items():w.writerow([int(t)+1,v['predicted_noise_mse'],v['zero_predictor_mse']])
    metrics={
       'dataset':'sklearn.datasets.load_digits (actual 8x8 handwritten-digit images)',
       'dataset_count':1797,'training_count':1257,'validation_count':270,'test_count':270,
       'split':'two-stage stratified fixed-size split; no index overlap',
       'seed':seed,'epochs_run':epochs,'checkpoint_epoch_selected_by_validation':best_epoch,
       'diffusion_timesteps':T,'beta_start':float(beta[0]),'beta_end':float(beta[-1]),
       'terminal_alpha_bar':float(abar[-1]),
       'model':'DDPM noise predictor MLP 66-256-256-64',
       'objective':'epsilon MSE for forward noising of real training images',
       'selected_validation_mse':val_scores,'independent_test_mse':test_scores,
       'independent_test_mean_epsilon_mse':float(measured.mean()),
       'independent_test_baseline_mean_epsilon_mse':float(baseline.mean()),
       'generated_sample_raw_value_range':raw_range,
       'test_index_sha256':hashlib.sha256(test_idx.astype('<i8').tobytes()).hexdigest(),
       'software':{'numpy':np.__version__,'scikit_learn':sklearn.__version__,'torch':torch.__version__},
       'limitations':'Small 8x8 DDPM demonstrator; no external generated-image quality certification, no DDIM comparison. Generated samples are model outputs, not photographs or observed images; not SOTA.'}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    return metrics


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path('ddpm_results'))
    parser.add_argument('--epochs',type=int,default=45)
    args=parser.parse_args()
    print(json.dumps(run(args.out,epochs=args.epochs),indent=2))
