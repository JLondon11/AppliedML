"""Listing 1.10 — Latitude-weighted RMSE for global weather forecast evaluation."""
from __future__ import annotations
import numpy as np

def latitude_weights(latitudes_deg):
    lat=np.asarray(latitudes_deg,dtype=float)
    w=np.cos(np.deg2rad(lat))
    return w/w.mean()

def latitude_weighted_rmse(y_true,y_pred,latitudes_deg):
    y_true=np.asarray(y_true,dtype=float); y_pred=np.asarray(y_pred,dtype=float)
    if y_true.shape!=y_pred.shape: raise ValueError("shape mismatch")
    if y_true.ndim<2: raise ValueError("expected latitude as penultimate dimension or latitude axis 0")
    w=latitude_weights(latitudes_deg)
    if y_true.shape[-2]!=len(w):
        raise ValueError("latitude count must match penultimate dimension")
    shape=[1]*y_true.ndim; shape[-2]=len(w)
    err2=(y_pred-y_true)**2
    return float(np.sqrt(np.mean(err2*w.reshape(shape))))

if __name__=="__main__":
    lats=np.array([-60,-30,0,30,60])
    truth=np.zeros((2,5,8))
    pred=np.ones_like(truth)
    print("weighted RMSE:",latitude_weighted_rmse(truth,pred,lats))
