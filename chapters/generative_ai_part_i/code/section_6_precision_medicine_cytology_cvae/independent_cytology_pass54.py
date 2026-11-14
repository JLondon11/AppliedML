"""Independent original WDBC real-data CVAE/classifier comparison.

Scope: educational tabular diagnostic. Not clinical inference or safety validation.
"""
import json, csv, hashlib, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEEDS=(11,23,37)
class CVAE(nn.Module):
    def __init__(self,d=30,z=10):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(d+2,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU())
        self.mu=nn.Linear(64,z);self.lv=nn.Linear(64,z)
        self.decoder=nn.Sequential(nn.Linear(z+2,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,d))
    def prior(self,z,y):
        return self.decoder(torch.cat((z,nn.functional.one_hot(y,2).float()),1))
    def forward(self,x,y):
        h=self.encoder(torch.cat((x,nn.functional.one_hot(y,2).float()),1))
        mu=self.mu(h);lv=self.lv(h).clamp(-6,5)
        z=mu+torch.randn_like(mu)*torch.exp(.5*lv)
        return self.prior(z,y),mu,lv

def run(out,epochs=35):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    ds=load_breast_cancer();X=ds.data.astype(np.float32);Y=ds.target.astype(np.int64)
    ids=np.arange(len(Y));train,other=train_test_split(ids,test_size=.4,random_state=20261109,stratify=Y)
    val,test=train_test_split(other,test_size=.5,random_state=20261109,stratify=Y[other])
    assert (len(train),len(val),len(test))==(341,114,114)
    assert not (set(train)&set(val) or set(train)&set(test) or set(val)&set(test))
    scaler=StandardScaler().fit(X[train]);sx=np.clip(scaler.transform(X),-8,8).astype('float32')
    trainx,trainy=sx[train],Y[train];testx,testy=sx[test],Y[test]
    metrics=[];predictions=[];samples=[];history=[]
    for seed in SEEDS:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        model=CVAE();opt=torch.optim.Adam(model.parameters(),lr=.001)
        x=torch.from_numpy(sx);y=torch.from_numpy(Y)
        best=float('inf');state=None;selected=0
        for epoch in range(1,epochs+1):
            model.train();order=torch.tensor(np.random.permutation(train));losses=[]
            for ix in order.split(64):
                rec,mu,lv=model(x[ix],y[ix]);recon=(rec-x[ix]).square().mean()
                kl=(-.5*(1+lv-mu.square()-lv.exp()).sum(1)).mean()
                loss=recon+.003*kl;opt.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                losses.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():
                va=x[val];yv=y[val];h=model.encoder(torch.cat((va,nn.functional.one_hot(yv,2).float()),1))
                vl=float((model.prior(model.mu(h),yv)-va).square().mean())
            history.append((seed,epoch,float(np.mean(losses)),vl))
            if vl<best:best=vl;selected=epoch;state={k:v.detach().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(state);model.eval()
        rng=np.random.default_rng(seed+1000);generated=[];labels=[]
        with torch.no_grad():
            for label in (0,1):
                n=int((trainy==label).sum())
                v=model.prior(torch.from_numpy(rng.standard_normal((n,10)).astype('float32')),torch.full((n,),label,dtype=torch.long)).numpy()
                generated.append(v);labels.extend([label]*n)
        generated=np.vstack(generated);labels=np.asarray(labels,dtype=np.int64)
        assert len(generated)==len(trainx)==341
        for i,(v,label) in enumerate(zip(generated,labels)):
            samples.append((seed,i,int(label),*map(float,v)))
        for arm in ('real_only','real_bootstrap','real_plus_cvae'):
            if arm=='real_only':bx,by=trainx,trainy
            elif arm=='real_bootstrap':
                extra=np.concatenate([rng.choice(np.flatnonzero(trainy==label),size=int((trainy==label).sum()),replace=True) for label in (0,1)])
                bx=np.vstack([trainx,trainx[extra]]);by=np.r_[trainy,trainy[extra]]
            else:bx=np.vstack([trainx,generated]);by=np.r_[trainy,labels]
            clf=LogisticRegression(C=1,max_iter=1000,random_state=seed).fit(bx,by)
            prob=clf.predict_proba(testx)[:,1];pred=(prob>=.5).astype(int)
            metrics.append(dict(seed=seed,arm=arm,n_train_effective=len(bx),accuracy=float(accuracy_score(testy,pred)),balanced_accuracy=float(balanced_accuracy_score(testy,pred)),macro_F1=float(f1_score(testy,pred,average='macro')),ROC_AUC=float(roc_auc_score(testy,prob)),selected_cvae_epoch=selected))
            predictions.extend((seed,arm,int(i),int(t),int(p),float(pr)) for i,(t,p,pr) in enumerate(zip(testy,pred,prob)))
        torch.save({'weights':state,'selected_epoch':selected,'seed':seed},out/f'cvae_seed{seed}.pt')
    for filename,rows,header in [
        ('metrics.csv',metrics,list(metrics[0].keys())),
        ('test_predictions.csv',predictions,['seed','arm','test_index','true_label','prediction','probability_benign']),
        ('prior_samples.csv',samples,['seed','index','class']+[f'feature_{i}' for i in range(30)]),
        ('training.csv',history,['seed','epoch','training_objective','validation_reconstruction_MSE'])]:
        with (out/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=header) if isinstance(rows[0],dict) else csv.writer(f)
            if isinstance(rows[0],dict):w.writeheader();w.writerows(rows)
            else:w.writerow(header);w.writerows(rows)
    arms=('real_only','real_bootstrap','real_plus_cvae')
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    colors=['#315c83','#7d8592','#b36e53']
    for i,arm in enumerate(arms):
        selected=[r for r in metrics if r['arm']==arm]
        for j,key in enumerate(('ROC_AUC','balanced_accuracy')):
            vals=np.array([r[key] for r in selected]);axes[j].scatter(i+np.array([-.10,0,.10]),vals,s=55,color=colors[i],alpha=.8)
            axes[j].hlines(vals.mean(),i-.2,i+.2,lw=2,color=colors[i])
    for j,title in enumerate(('ROC AUC','Balanced accuracy')):
        axes[j].set(xticks=range(3),xticklabels=['Real only','Real repeat','Real + CVAE'],ylim=(.85,1.01),ylabel=title,title=f'Held-out original cases · {title}')
        axes[j].grid(axis='y',alpha=.15)
    fig.suptitle('WDBC original test-set classification; same 114 genuine cases across all seeds',fontsize=10)
    for ext in ('png','pdf'):fig.savefig(out/('wdbc_original_test_ablation.'+ext),dpi=210,bbox_inches='tight')
    plt.close(fig)
    grouped={arm:{key:{'mean':float(np.mean([r[key] for r in metrics if r['arm']==arm])),'sample_sd':float(np.std([r[key] for r in metrics if r['arm']==arm],ddof=1))} for key in ('ROC_AUC','balanced_accuracy','macro_F1')} for arm in arms}
    manifest=dict(dataset='Wisconsin Diagnostic Breast Cancer original cytology measurements via sklearn',n_source=len(Y),features=30,sha256=hashlib.sha256(X.tobytes()+Y.tobytes()).hexdigest(),training=341,validation=114,testing=114,seeds=list(SEEDS),epochs=epochs,split_train=train.tolist(),split_validation=val.tolist(),split_test=test.tolist(),measured_metrics=grouped,prior_generated_per_seed=341,scope='single-modality historical cytology records; offline augmentation only',limitations=['No independent hospital or patient cohort','Same 114 real held-out cases across all seeds','No clinical safety, privacy, subgroup or calibration validation','Generated vectors are not genuine observations'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(grouped,indent=2))
    return manifest
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('results'));p.add_argument('--epochs',type=int,default=35)
    a=p.parse_args();run(a.output,a.epochs)
