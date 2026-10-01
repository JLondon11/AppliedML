"""Listing 1.8 — Cost-sensitive threshold selection for a fraud classifier."""
from __future__ import annotations
import numpy as np

def select_cost_optimal_threshold(y_true,y_prob,cost_false_negative=500.0,
                                  cost_false_positive=5.0,thresholds=None):
    y_true=np.asarray(y_true,dtype=int); y_prob=np.asarray(y_prob,dtype=float)
    if thresholds is None: thresholds=np.linspace(.01,.99,99)
    rows=[]
    for t in thresholds:
        pred=(y_prob>=t).astype(int)
        fn=int(((y_true==1)&(pred==0)).sum())
        fp=int(((y_true==0)&(pred==1)).sum())
        cost=fn*cost_false_negative+fp*cost_false_positive
        rows.append((float(t),float(cost),fn,fp))
    return min(rows,key=lambda x:x[1]),rows

if __name__=="__main__":
    rng=np.random.default_rng(42)
    y=np.r_[np.zeros(990,dtype=int),np.ones(10,dtype=int)]
    p=np.clip(.05+rng.normal(0,.03,len(y))+y*.65,0,1)
    best,_=select_cost_optimal_threshold(y,p)
    print({"threshold":best[0],"cost":best[1],"false_negatives":best[2],"false_positives":best[3]})
