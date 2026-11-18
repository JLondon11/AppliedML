"""German Credit logistic regression with true SGD, Momentum, AdamW, and L-BFGS."""
import argparse, json, time
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, log_loss

def prepare(seed):
    ds=fetch_openml('credit-g',version=1,as_frame=True,parser='auto')
    X=ds.data; y=(ds.target.astype(str)=='good').astype('float32').to_numpy()
    num=X.select_dtypes(include=['number']).columns.tolist(); cat=[c for c in X.columns if c not in num]
    pre=ColumnTransformer([
      ('num',Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]),num),
      ('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=seed)
    Xtr=pre.fit_transform(Xtr).astype('float32'); Xte=pre.transform(Xte).astype('float32')
    return Xtr,Xte,ytr,yte

def train(name,Xtr,Xte,ytr,yte,seed,steps=1000):
    import torch
    torch.manual_seed(seed); x=torch.tensor(Xtr); y=torch.tensor(ytr)[:,None]; xt=torch.tensor(Xte)
    model=torch.nn.Linear(x.shape[1],1); criterion=torch.nn.BCEWithLogitsLoss()
    if name=='sgd': opt=torch.optim.SGD(model.parameters(),lr=.05,weight_decay=1e-4)
    elif name=='momentum': opt=torch.optim.SGD(model.parameters(),lr=.05,momentum=.9,weight_decay=1e-4)
    elif name=='adamw': opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=1e-4)
    elif name=='lbfgs': opt=torch.optim.LBFGS(model.parameters(),lr=1.0,max_iter=steps,line_search_fn='strong_wolfe')
    else: raise ValueError(name)
    t0=time.perf_counter()
    if name=='lbfgs':
        def closure(): opt.zero_grad(); loss=criterion(model(x),y); loss.backward(); return loss
        opt.step(closure)
    else:
        for _ in range(steps): opt.zero_grad(); loss=criterion(model(x),y); loss.backward(); opt.step()
    runtime=time.perf_counter()-t0
    opt.zero_grad(); loss=criterion(model(x),y); loss.backward(); grad=np.sqrt(sum(float((p.grad.detach()**2).sum()) for p in model.parameters()))
    with torch.no_grad(): prob=torch.sigmoid(model(xt)).squeeze(1).numpy()
    return {'optimizer':name,'roc_auc':float(roc_auc_score(yte,prob)),'log_loss':float(log_loss(yte,prob)),'runtime_s':runtime,'gradient_norm':grad}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,default=7); ap.add_argument('--steps',type=int,default=1000); ap.add_argument('--out',default='ce01_results.json'); a=ap.parse_args()
    Xtr,Xte,ytr,yte=prepare(a.seed)
    results=[train(n,Xtr,Xte,ytr,yte,a.seed,a.steps) for n in ['sgd','momentum','adamw','lbfgs']]
    open(a.out,'w').write(json.dumps(results,indent=2)); print(json.dumps(results,indent=2))
if __name__=='__main__': main()
