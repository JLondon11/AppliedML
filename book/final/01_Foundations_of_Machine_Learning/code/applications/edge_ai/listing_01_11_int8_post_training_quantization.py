"""Listing 1.11 — INT8 post-training quantization of a weight tensor."""
from __future__ import annotations
import numpy as np

def quantize_uint8_affine(weights):
    w=np.asarray(weights,dtype=np.float32)
    w_min=float(w.min()); w_max=float(w.max())
    if np.isclose(w_max,w_min):
        return np.zeros_like(w,dtype=np.uint8),1.0,0
    scale=(w_max-w_min)/255.0
    zero_point=int(np.clip(round(-w_min/scale),0,255))
    q=np.round(w/scale+zero_point)
    q=np.clip(q,0,255).astype(np.uint8)
    return q,float(scale),zero_point

def dequantize_uint8_affine(q,scale,zero_point):
    return scale*(np.asarray(q,dtype=np.float32)-zero_point)

if __name__=="__main__":
    rng=np.random.default_rng(42)
    w=rng.normal(0,.25,size=(128,64)).astype(np.float32)
    q,s,z=quantize_uint8_affine(w)
    w_hat=dequantize_uint8_affine(q,s,z)
    print({"float_bytes":w.nbytes,"quantized_bytes":q.nbytes,
           "rmse":float(np.sqrt(np.mean((w-w_hat)**2)))})
