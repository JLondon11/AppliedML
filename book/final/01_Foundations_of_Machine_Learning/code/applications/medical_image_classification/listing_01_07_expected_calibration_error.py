"""Listing 1.7 — Expected calibration error for a binary classifier."""
from __future__ import annotations
import numpy as np

def expected_calibration_error(y_true,y_prob,n_bins=10):
    y_true=np.asarray(y_true,dtype=int); y_prob=np.asarray(y_prob,dtype=float)
    if len(y_true)!=len(y_prob): raise ValueError("length mismatch")
    edges=np.linspace(0,1,n_bins+1); ece=0.0; n=len(y_true)
    for i in range(n_bins):
        left,right=edges[i],edges[i+1]
        mask=(y_prob>=left)&((y_prob<right) if i<n_bins-1 else (y_prob<=right))
        if not mask.any(): continue
        confidence=float(y_prob[mask].mean())
        accuracy=float(y_true[mask].mean())
        ece+=(mask.sum()/n)*abs(confidence-accuracy)
    return float(ece)

if __name__=="__main__":
    y=np.array([0,0,1,1,1,0,1,0])
    p=np.array([.05,.2,.65,.9,.8,.4,.55,.3])
    print("ECE:",expected_calibration_error(y,p,n_bins=4))
