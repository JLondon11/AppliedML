"""Listing 1.1 — Minimal AdamW optimizer update.

Full runnable companion for the manuscript excerpt.
"""
from __future__ import annotations
import numpy as np

def adamw_step(theta, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999,
               eps=1e-8, weight_decay=0.01):
    theta=np.asarray(theta,dtype=float)
    grad=np.asarray(grad,dtype=float)
    m=np.asarray(m,dtype=float); v=np.asarray(v,dtype=float)
    m=beta1*m+(1-beta1)*grad
    v=beta2*v+(1-beta2)*(grad**2)
    m_hat=m/(1-beta1**t)
    v_hat=v/(1-beta2**t)
    theta=theta-lr*m_hat/(np.sqrt(v_hat)+eps)
    theta=theta-lr*weight_decay*theta
    return theta,m,v

def demo(seed=42, steps=200):
    rng=np.random.default_rng(seed)
    theta=rng.normal(size=4); m=np.zeros_like(theta); v=np.zeros_like(theta)
    target=np.array([1.0,-2.0,0.5,3.0])
    for t in range(1,steps+1):
        grad=theta-target
        theta,m,v=adamw_step(theta,grad,m,v,t,lr=0.05,weight_decay=1e-3)
    return theta

if __name__=="__main__":
    print("final parameters:", np.round(demo(),6))
