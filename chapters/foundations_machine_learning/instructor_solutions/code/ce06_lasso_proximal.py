"""Proximal-gradient LASSO compared with sklearn coordinate descent."""
import argparse, json, numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

def soft(v,t): return np.sign(v)*np.maximum(np.abs(v)-t,0)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--lam',type=float,default=.05); ap.add_argument('--iters',type=int,default=10000); ap.add_argument('--out',default='ce06_results.json'); a=ap.parse_args()
    X,y=load_diabetes(return_X_y=True); X=StandardScaler().fit_transform(X); y=y-y.mean(); n=len(y)
    L=np.linalg.norm(X,2)**2/n; step=1/L; w=np.zeros(X.shape[1])
    for _ in range(a.iters): w=soft(w-step*(X.T@(X@w-y)/n),step*a.lam)
    cd=Lasso(alpha=a.lam,max_iter=10000,tol=1e-10,fit_intercept=False).fit(X,y)
    obj=lambda z:.5*np.mean((X@z-y)**2)+a.lam*np.abs(z).sum()
    r={'prox_objective':obj(w),'cd_objective':obj(cd.coef_),'coef_l2_difference':float(np.linalg.norm(w-cd.coef_)),'prox_nonzero':int((np.abs(w)>1e-8).sum())}
    open(a.out,'w').write(json.dumps(r,indent=2)); print(r)
if __name__=='__main__': main()
