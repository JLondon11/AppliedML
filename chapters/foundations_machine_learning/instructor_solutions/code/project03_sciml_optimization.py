"""Scientific-ML optimization utilities: adaptive weighting, hybrid AdamW/L-BFGS, residual refinement."""
import torch

def adaptive_weights(losses,eps=1e-12):
    inv=torch.stack([1/(x.detach()+eps) for x in losses]); return inv/inv.sum()*len(losses)

def refine_collocation(points,residual_fn,k):
    with torch.enable_grad(): score=residual_fn(points).detach().abs().flatten()
    return points[torch.topk(score,min(k,len(score))).indices]

def hybrid_optimize(model,loss_fn,adam_steps=1000,lbfgs_steps=100):
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    for _ in range(adam_steps): opt.zero_grad(); L=loss_fn(); L.backward(); opt.step()
    lb=torch.optim.LBFGS(model.parameters(),max_iter=lbfgs_steps,line_search_fn='strong_wolfe')
    def closure(): lb.zero_grad(); L=loss_fn(); L.backward(); return L
    lb.step(closure); return float(loss_fn().detach())

if __name__=='__main__': print('Import these utilities into a task-specific PINN model.')
