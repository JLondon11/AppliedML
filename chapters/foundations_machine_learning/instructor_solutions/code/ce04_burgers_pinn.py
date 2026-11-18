"""PINN for viscous Burgers equation with AdamW and L-BFGS refinement."""
import argparse, json

def main():
    import torch
    from torch import nn
    ap=argparse.ArgumentParser(); ap.add_argument('--adam-steps',type=int,default=3000); ap.add_argument('--lbfgs-steps',type=int,default=300); ap.add_argument('--out',default='ce04_results.json'); a=ap.parse_args()
    torch.manual_seed(7); nu=0.01/3.141592653589793
    net=nn.Sequential(nn.Linear(2,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))
    def residual(x,t):
        z=torch.cat([x,t],1).requires_grad_(True); u=net(z)
        g=torch.autograd.grad(u,z,torch.ones_like(u),create_graph=True)[0]; ux,ut=g[:,0:1],g[:,1:2]
        uxx=torch.autograd.grad(ux,z,torch.ones_like(ux),create_graph=True)[0][:,0:1]
        return ut+u*ux-nu*uxx
    def loss():
        x=torch.rand(2000,1)*2-1; t=torch.rand(2000,1); r=residual(x,t)
        xi=torch.rand(512,1)*2-1; ti=torch.zeros_like(xi); ui=-torch.sin(torch.pi*xi)
        xb=torch.cat([-torch.ones(256,1),torch.ones(256,1)],0); tb=torch.rand(512,1)
        return (r*r).mean()+((net(torch.cat([xi,ti],1))-ui)**2).mean()+((net(torch.cat([xb,tb],1)))**2).mean()
    opt=torch.optim.AdamW(net.parameters(),lr=1e-3)
    for _ in range(a.adam_steps): opt.zero_grad(); L=loss(); L.backward(); opt.step()
    lb=torch.optim.LBFGS(net.parameters(),max_iter=a.lbfgs_steps,line_search_fn='strong_wolfe')
    def closure(): lb.zero_grad(); L=loss(); L.backward(); return L
    lb.step(closure)
    final=float(loss().detach()); result={'final_composite_loss':final}
    open(a.out,'w').write(json.dumps(result,indent=2)); print(result)
if __name__=='__main__': main()
