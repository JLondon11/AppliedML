from pathlib import Path
import time, json, argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, log_loss
import matplotlib.pyplot as plt

COLORS={'SGD':'#355C7D','Momentum':'#4F7C6E','AdamW':'#B26E3B','L-BFGS':'#7A6AA6'}
SEEDS=[7,19,41]

def sigmoid(z):
    z=np.clip(z,-40,40)
    return 1/(1+np.exp(-z))

def loss_grad(theta,X,y,wd=1e-4):
    w,b=theta[:-1],theta[-1]
    p=sigmoid(X@w+b)
    eps=1e-12
    loss=-(y*np.log(p+eps)+(1-y)*np.log(1-p+eps)).mean()+0.5*wd*np.dot(w,w)
    e=(p-y)/len(y)
    grad=np.r_[X.T@e+wd*w,e.sum()]
    return float(loss),grad

def prep(seed=7, standardized=True, scale_factor=1.0):
    d=load_breast_cancer()
    X=d.data.astype(float); y=d.target.astype(float)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=seed)
    if standardized:
        sc=StandardScaler().fit(Xtr)
        Xtr=sc.transform(Xtr); Xte=sc.transform(Xte)
    else:
        rms=np.sqrt(np.mean(Xtr**2,axis=0)); rms[rms==0]=1
        Xtr=Xtr/rms; Xte=Xte/rms
        Xtr[:,:10]*=scale_factor; Xte[:,:10]*=scale_factor
    return Xtr,Xte,ytr,yte

def run_first_order(name,X,y,max_steps=250,lr=.05,wd=1e-4):
    th=np.zeros(X.shape[1]+1); m=np.zeros_like(th); v=np.zeros_like(th)
    losses=[]; t0=time.perf_counter(); prev=None
    for step in range(1,max_steps+1):
        loss,g=loss_grad(th,X,y,wd=wd); losses.append(loss)
        if name=='SGD':
            th-=lr*g
        elif name=='Momentum':
            m=.9*m+g; th-=lr*m
        elif name=='AdamW':
            b1,b2=.9,.999
            m=b1*m+(1-b1)*g; v=b2*v+(1-b2)*(g*g)
            mh=m/(1-b1**step); vh=v/(1-b2**step)
            th-=lr*mh/(np.sqrt(vh)+1e-8)
            th[:-1]*=(1-lr*wd)
        if prev is not None and abs(prev-loss)<1e-7 and step>15:
            break
        prev=loss
    return th,losses,time.perf_counter()-t0

def run_lbfgs(X,y,max_steps=250,wd=1e-4):
    hist=[]; t0=time.perf_counter()
    def fun(th):
        l,g=loss_grad(th,X,y,wd)
        return l,g
    def cb(th):
        hist.append(loss_grad(th,X,y,wd)[0])
    res=minimize(fun,np.zeros(X.shape[1]+1),method='L-BFGS-B',jac=True,callback=cb,
                 options={'maxiter':max_steps,'ftol':1e-12,'gtol':1e-8,'maxls':30})
    if not hist:
        hist=[float(res.fun)]
    return res.x,hist,time.perf_counter()-t0

def predict(th,X):
    return sigmoid(X@th[:-1]+th[-1])

def optimizer_benchmark(out):
    runs=[]; traces={}
    for seed in SEEDS:
        Xtr,Xte,ytr,yte=prep(seed,True)
        for name in ['SGD','Momentum','AdamW','L-BFGS']:
            if name=='SGD':
                th,losses,elapsed=run_first_order(name,Xtr,ytr,lr=.08)
            elif name=='Momentum':
                th,losses,elapsed=run_first_order(name,Xtr,ytr,lr=.05)
            elif name=='AdamW':
                th,losses,elapsed=run_first_order(name,Xtr,ytr,lr=.03)
            else:
                th,losses,elapsed=run_lbfgs(Xtr,ytr)
            p=predict(th,Xte); pred=(p>=.5).astype(int)
            runs.append({'optimizer':name,'seed':seed,'steps':len(losses),'train_loss':losses[-1],
                         'balanced_accuracy':balanced_accuracy_score(yte,pred),
                         'roc_auc':roc_auc_score(yte,p),
                         'test_log_loss':log_loss(yte,p,labels=[0,1]),'elapsed_s':elapsed})
            traces[(name,seed)]=losses
    df=pd.DataFrame(runs)
    df.to_csv(out/'optimizer_benchmark_runs.csv',index=False)
    sm=df.groupby('optimizer').agg(
        steps_mean=('steps','mean'),steps_sd=('steps','std'),
        train_loss_mean=('train_loss','mean'),
        balanced_accuracy_mean=('balanced_accuracy','mean'),
        balanced_accuracy_sd=('balanced_accuracy','std'),
        roc_auc_mean=('roc_auc','mean'),roc_auc_sd=('roc_auc','std'),
        test_log_loss_mean=('test_log_loss','mean'),
        elapsed_s_mean=('elapsed_s','mean')
    ).reset_index()
    sm.to_csv(out/'optimizer_benchmark_table.csv',index=False)

    fig,axs=plt.subplots(1,2,figsize=(10.2,4.15))
    for name in ['SGD','Momentum','AdamW','L-BFGS']:
        rr=[traces[(name,s)] for s in SEEDS]
        n=min(120,max(map(len,rr))); grid=np.arange(1,n+1); vals=[]
        for y in rr:
            x=np.arange(1,len(y)+1)
            vals.append(np.interp(grid,x,y,left=y[0],right=y[-1]))
        vals=np.vstack(vals); m=vals.mean(0); sd=vals.std(0)
        axs[0].plot(grid,m,label=name,lw=2,color=COLORS[name])
        axs[0].fill_between(grid,m-sd,m+sd,alpha=.14,color=COLORS[name],linewidth=0)
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Optimization step')
    axs[0].set_ylabel('Training binary cross-entropy')
    axs[0].legend(frameon=False,ncol=2,fontsize=8)

    order=['SGD','Momentum','AdamW','L-BFGS']; si=sm.set_index('optimizer'); x=np.arange(4)
    axs[1].bar(x,[si.loc[n,'balanced_accuracy_mean'] for n in order],
               yerr=[si.loc[n,'balanced_accuracy_sd'] for n in order],
               capsize=3,color=[COLORS[n] for n in order],width=.64)
    axs[1].set_xticks(x,order,rotation=18,ha='right')
    axs[1].set_ylabel('Held-out balanced accuracy')
    axs[1].set_ylim(.85,1.0)
    for i,ax in enumerate(axs):
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(direction='out')
        ax.text(.5,-.23,f'({chr(97+i)})',transform=ax.transAxes,ha='center',va='top',fontsize=10)
    fig.tight_layout(w_pad=2.1)
    for ext in ['png','svg','pdf']:
        fig.savefig(out/f'ch01_optimizer_selection_benchmark.{ext}',
                    dpi=300 if ext=='png' else None,bbox_inches='tight')
    plt.close(fig)
    return sm

def conditioning(out):
    rows=[]
    for sf in [1,3,10,30,100]:
        Xtr,Xte,ytr,yte=prep(7,False,sf)
        gram=(Xtr.T@Xtr)/len(Xtr)+1e-4*np.eye(Xtr.shape[1])
        cond=float(np.linalg.cond(gram))
        for name in ['SGD','L-BFGS']:
            if name=='SGD':
                th,losses,elapsed=run_first_order('SGD',Xtr,ytr,max_steps=400,
                                                  lr=.005/max(1,sf/3))
            else:
                th,losses,elapsed=run_lbfgs(Xtr,ytr,max_steps=400)
            p=predict(th,Xte)
            hit=np.where(np.asarray(losses)<.12)[0]
            steps=int(hit[0]+1) if len(hit) else 400
            rows.append({'scale_factor':sf,'condition_number':cond,'optimizer':name,
                         'steps_to_threshold':steps,'threshold_reached':bool(len(hit)),
                         'final_train_loss':losses[-1],
                         'balanced_accuracy':balanced_accuracy_score(yte,(p>=.5).astype(int)),
                         'elapsed_s':elapsed})
    df=pd.DataFrame(rows)
    df.to_csv(out/'conditioning_sensitivity_table.csv',index=False)

    fig,axs=plt.subplots(1,2,figsize=(10.2,4.15))
    for name in ['SGD','L-BFGS']:
        q=df[df.optimizer==name].sort_values('condition_number')
        axs[0].plot(q.condition_number,q.steps_to_threshold,marker='o',lw=2,label=name,color=COLORS[name])
        axs[1].plot(q.condition_number,q.balanced_accuracy,marker='o',lw=2,label=name,color=COLORS[name])
    axs[0].set_xscale('log'); axs[0].set_yscale('log')
    axs[0].set_xlabel('Condition number of regularized feature Gram matrix')
    axs[0].set_ylabel('Optimization steps to loss threshold (cap = 400)')
    axs[0].legend(frameon=False)
    axs[1].set_xscale('log')
    axs[1].set_xlabel('Condition number of regularized feature Gram matrix')
    axs[1].set_ylabel('Held-out balanced accuracy')
    axs[1].set_ylim(.45,1.0)
    for i,ax in enumerate(axs):
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(direction='out')
        ax.text(.5,-.23,f'({chr(97+i)})',transform=ax.transAxes,ha='center',va='top',fontsize=10)
    fig.tight_layout(w_pad=2.2)
    for ext in ['png','svg','pdf']:
        fig.savefig(out/f'ch01_optimizer_conditioning_sensitivity.{ext}',
                    dpi=300 if ext=='png' else None,bbox_inches='tight')
    plt.close(fig)
    return df

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='book/final/01_Foundations_of_Machine_Learning/figures')
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    sm=optimizer_benchmark(out); cd=conditioning(out)
    meta={
        'dataset':'sklearn.datasets.load_breast_cancer (Wisconsin Diagnostic Breast Cancer)',
        'seeds':SEEDS,
        'methods':'NumPy first-order optimizers and SciPy L-BFGS-B on the same regularized logistic objective'
    }
    (out/'ch01_optimizer_evidence_provenance.json').write_text(json.dumps(meta,indent=2))
    print(sm.to_string(index=False)); print(cd.to_string(index=False))

if __name__=='__main__':
    main()
