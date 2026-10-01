"""Listing 1.14 — NDVI computation from multispectral imagery."""
from __future__ import annotations
import numpy as np

def compute_ndvi(nir_band,red_band,epsilon=1e-8):
    nir=np.asarray(nir_band,dtype=np.float32)
    red=np.asarray(red_band,dtype=np.float32)
    if nir.shape!=red.shape: raise ValueError("band shapes must match")
    ndvi=(nir-red)/(nir+red+epsilon)
    return np.clip(ndvi,-1.0,1.0)

if __name__=="__main__":
    nir=np.array([[.8,.6],[.3,.2]],dtype=np.float32)
    red=np.array([[.2,.3],[.25,.2]],dtype=np.float32)
    print(compute_ndvi(nir,red))
