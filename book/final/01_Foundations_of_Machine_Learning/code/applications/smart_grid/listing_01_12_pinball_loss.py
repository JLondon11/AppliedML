"""Listing 1.12 — Pinball (quantile) loss for probabilistic load forecasting."""
from __future__ import annotations
import numpy as np

def pinball_loss(y_true,y_pred_quantile,alpha):
    if not 0<alpha<1: raise ValueError("alpha must be in (0,1)")
    y=np.asarray(y_true,dtype=float); q=np.asarray(y_pred_quantile,dtype=float)
    e=y-q
    return float(np.mean(np.where(e>=0,alpha*e,(alpha-1)*e)))

def mean_pinball_across_quantiles(y_true,quantile_preds,alphas):
    qp=np.asarray(quantile_preds,dtype=float)
    if qp.shape[1]!=len(alphas): raise ValueError("quantile count mismatch")
    return float(np.mean([pinball_loss(y_true,qp[:,i],a) for i,a in enumerate(alphas)]))

if __name__=="__main__":
    y=np.array([10,12,15,18,20],dtype=float)
    preds=np.column_stack([y-2,y,y+2])
    print(mean_pinball_across_quantiles(y,preds,[.1,.5,.9]))
