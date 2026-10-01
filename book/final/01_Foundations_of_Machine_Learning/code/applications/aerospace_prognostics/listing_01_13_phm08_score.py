"""Listing 1.13 — Asymmetric PHM08 prognostic scoring function."""
from __future__ import annotations
import numpy as np

def phm_score(y_true_rul,y_pred_rul):
    y_true=np.asarray(y_true_rul,dtype=float); y_pred=np.asarray(y_pred_rul,dtype=float)
    if y_true.shape!=y_pred.shape: raise ValueError("shape mismatch")
    error=y_pred-y_true
    scores=np.where(error<0,np.exp(-error/13.0)-1.0,np.exp(error/10.0)-1.0)
    return float(np.sum(scores))

if __name__=="__main__":
    truth=np.array([100,80,60,40,20],dtype=float)
    early=truth-5; late=truth+5
    print({"early_score":phm_score(truth,early),"late_score":phm_score(truth,late)})
