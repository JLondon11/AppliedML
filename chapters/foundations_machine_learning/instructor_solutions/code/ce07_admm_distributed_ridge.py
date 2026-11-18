"""Consensus ADMM for partitioned ridge regression."""
import argparse, json, numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--workers',type=int,default=4); ap.add_argument('--lam',type=float,default=1.0); ap.add_argument('--rho',type=float,default=10.0); ap.add_argument('--iters',type=int,default=300); ap.add_argument('--out',default='ce07_results.json'); a=ap.parse_args()
    X,y=load_diabetes(return_X_y=True); X=StandardScaler().fit_transform(X); y=y-y.mean(); parts=np.array_split(np.arange(len(y)),a.workers); d=X.shape[1]
    z=np.zeros(d); u=[np.zeros(d) for _ in parts]; xs=[np.zeros(d) for _ in parts]
    hist=[]
    for k in range(a.iters):
        for j,idx in enumerate(parts):
            Xi,yi=X[idx],y[idx]; A=Xi.T@Xi+(a.rho+a.lam/a.workers)*np.eye(d); b=Xi.T@yi+a.rho*(z-u[j]); xs[j]=np.linalg.solve(A,b)
        z_old=z.copy(); z=np.mean([xs[j]+u[j] for j in range(a.workers)],axis=0)
        for j in range(a.workers): u[j]+=xs[j]-z
        pr=np.sqrt(sum(np.linalg.norm(x-z)**2 for x in xs)); dr=a.rho*np.sqrt(a.workers)*np.linalg.norm(z-z_old); hist.append((pr,dr))
    central=np.linalg.solve(X.T@X+a.lam*np.eye(d),X.T@y)
    r={'distance_to_central':float(np.linalg.norm(z-central)),'primal_residual':hist[-1][0],'dual_residual':hist[-1][1]}
    open(a.out,'w').write(json.dumps(r,indent=2)); print(r)
if __name__=='__main__': main()
