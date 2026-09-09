"""Reproduce Scientific AI Figure 11 with an actually trained CPU PINN."""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

torch.manual_seed(11); np.random.seed(11); torch.set_num_threads(2)
nu = 0.1
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),
                                 nn.Linear(32,32),nn.Tanh(),nn.Linear(32,1))
    def forward(self,z): return self.net(z)
m = Net(); opt = torch.optim.Adam(m.parameters(), lr=1.2e-3)
Nf, Ni, Nb = 1000, 140, 140
xf, tf = torch.rand(Nf,1), torch.rand(Nf,1)
xi, ti = torch.rand(Ni,1), torch.zeros(Ni,1)
tb = torch.rand(Nb,1); x0, x1 = torch.zeros_like(tb), torch.ones_like(tb)
def residual(x,t,cg=True):
    x=x.detach().clone().requires_grad_(True); t=t.detach().clone().requires_grad_(True)
    u=m(torch.cat([x,t],1))
    ut=torch.autograd.grad(u,t,torch.ones_like(u),create_graph=True,retain_graph=True)[0]
    ux=torch.autograd.grad(u,x,torch.ones_like(u),create_graph=True,retain_graph=True)[0]
    uxx=torch.autograd.grad(ux,x,torch.ones_like(ux),create_graph=cg,retain_graph=cg)[0]
    return ut-nu*uxx
for _ in range(2400):
    opt.zero_grad(); lp=(residual(xf,tf)**2).mean()
    li=((m(torch.cat([xi,ti],1))-torch.sin(torch.pi*xi))**2).mean()
    lb=(m(torch.cat([x0,tb],1))**2).mean()+(m(torch.cat([x1,tb],1))**2).mean()
    loss=lp+8*li+4*lb; loss.backward(); opt.step()
nx,nt=120,100; x=np.linspace(0,1,nx); t=np.linspace(0,1,nt); X,T=np.meshgrid(x,t)
with torch.no_grad():
    Up=m(torch.tensor(np.c_[X.ravel(),T.ravel()],dtype=torch.float32)).numpy().reshape(nt,nx)
Uref=np.exp(-nu*np.pi**2*T)*np.sin(np.pi*X); E=np.abs(Up-Uref)
R=np.empty(X.size)
for s in range(0,X.size,2000):
    e=min(s+2000,X.size)
    R[s:e]=np.abs(residual(torch.tensor(X.ravel()[s:e,None],dtype=torch.float32),
                           torch.tensor(T.ravel()[s:e,None],dtype=torch.float32),False).detach().numpy().ravel())
R=R.reshape(nt,nx)
plt.rcParams.update({"font.family":"DejaVu Serif","font.size":9,"axes.linewidth":.75})
fig=plt.figure(figsize=(12.2,5.9)); gs=GridSpec(2,2,figure=fig,height_ratios=[1,.085],wspace=.28,hspace=.11)
for j,(A,title,clab) in enumerate([
    (np.log10(R+1e-8),"Equation-residual magnitude",r"$\log_{10}|u_t-\nu u_{xx}|$"),
    (np.log10(E+1e-8),"Field error relative to reference",r"$\log_{10}|u_\theta-u_{\rm ref}|$")]):
    ax=fig.add_subplot(gs[0,j]); im=ax.imshow(A,origin="lower",extent=[0,1,0,1],aspect="auto",cmap="cividis",interpolation="bilinear")
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$t$"); ax.set_title(title); fig.colorbar(im,ax=ax,pad=.025,label=clab)
for j,L in enumerate(["(a)","(b)"]):
    ax=fig.add_subplot(gs[1,j]); ax.axis("off"); ax.text(.5,.22,L,ha="center",va="center",fontsize=10.5)
fig.subplots_adjust(bottom=.12,top=.90,left=.08,right=.96)
fig.savefig("../figures/figure_11_pinn_residual_field_error.svg",bbox_inches="tight",facecolor="white")
print("relative L2", np.linalg.norm(Up-Uref)/np.linalg.norm(Uref))
print("RMSE", np.sqrt(np.mean((Up-Uref)**2)))
print("mean |residual|", R.mean())
