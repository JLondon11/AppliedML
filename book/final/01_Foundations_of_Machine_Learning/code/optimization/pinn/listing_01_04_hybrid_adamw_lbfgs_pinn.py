"""Listing 1.4 — Hybrid AdamW-then-L-BFGS PINN training strategy.

Solves u''(x) + pi^2 sin(pi x)=0 on x in [0,1], u(0)=u(1)=0.
"""
from __future__ import annotations
import math, torch
from torch import nn

torch.manual_seed(42)

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(1,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,1))
    def forward(self,x): return self.net(x)

def pinn_loss(model,n=64):
    x=torch.linspace(0,1,n).reshape(-1,1).requires_grad_(True)
    u=model(x)
    du=torch.autograd.grad(u,x,torch.ones_like(u),create_graph=True)[0]
    d2u=torch.autograd.grad(du,x,torch.ones_like(du),create_graph=True)[0]
    residual=d2u+(math.pi**2)*torch.sin(math.pi*x)
    bc=model(torch.tensor([[0.0],[1.0]]))
    return (residual**2).mean()+20.0*(bc**2).mean()

def train_pinn_hybrid(model,adamw_epochs=400,lbfgs_iters=80):
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    for _ in range(adamw_epochs):
        optimizer.zero_grad(); loss=pinn_loss(model); loss.backward(); optimizer.step()
    lbfgs=torch.optim.LBFGS(model.parameters(),max_iter=lbfgs_iters,line_search_fn="strong_wolfe")
    def closure():
        lbfgs.zero_grad(); loss=pinn_loss(model); loss.backward(); return loss
    lbfgs.step(closure)
    return model

if __name__=="__main__":
    model=train_pinn_hybrid(PINN())
    print("final loss:", float(pinn_loss(model).detach()))
