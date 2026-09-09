"""Reproduce Scientific AI Figure 2 from explicit mathematical models."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
np.random.seed(7)
fig=plt.figure(figsize=(15,7.6)); gs=GridSpec(2,4,figure=fig,height_ratios=[1,.055],hspace=.12,wspace=.48)
ax1=fig.add_subplot(gs[0,0])
x=np.linspace(0,2*np.pi,26); t=np.linspace(0,2*np.pi,22); X,T=np.meshgrid(x,t)
U=np.sin(X-.75*T)+.28*np.sin(2*X+.35*T)
im=ax1.pcolormesh(x,t,U,shading="auto",cmap="coolwarm",vmin=-1.3,vmax=1.3)
ax1.scatter(X[::3,::3].ravel(),T[::3,::3].ravel(),s=9,facecolors="none",edgecolors="k",linewidths=.45)
ax1.set(xlabel=r"$x$",ylabel=r"$t$",title="Observation and measurement")
fig.colorbar(im,ax=ax1,fraction=.046,pad=.03,label=r"$u(x,t)$")
ax2=fig.add_subplot(gs[0,1]); y=np.linspace(-1,1,400); u=1-y*y
ax2.plot(u,y,lw=2); ax2.axhline(-1,lw=.8); ax2.axhline(1,lw=.8)
ax2.set(xlim=(0,1.05),ylim=(-1.15,1.15),xlabel=r"$u/U_{\max}$",ylabel=r"$y/h$",title="Mechanistic representation")
ax2.text(.04,.96,r"$u/U_{\max}=1-(y/h)^2$",transform=ax2.transAxes,ha="left",va="top")
def f(q):
    xx,yy,zz=q
    return np.array([10*(yy-xx),xx*(28-zz)-yy,xx*yy-(8/3)*zz])
dt=.005;n=6000;q=np.empty((n,3));q[0]=[1,1,1]
for k in range(n-1):
    k1=f(q[k]);k2=f(q[k]+dt*k1/2);k3=f(q[k]+dt*k2/2);k4=f(q[k]+dt*k3)
    q[k+1]=q[k]+dt*(k1+2*k2+2*k3+k4)/6
ax3=fig.add_subplot(gs[0,2],projection="3d");ax3.plot(q[:,0],q[:,1],q[:,2],lw=.55)
ax3.set(xlabel=r"$x$",ylabel=r"$y$",zlabel=r"$z$",title="Numerical simulation");ax3.view_init(25,-55)
ax4=fig.add_subplot(gs[0,3]);xs=np.linspace(-2.5,2.5,70)
truth=np.exp(-.55*xs**2)*np.cos(2.1*xs);idx=np.arange(0,len(xs),7);xo=xs[idx];yo=truth[idx]+np.random.normal(0,.045,len(idx))
deg=8;Phi=np.vander(xo,deg+1,increasing=True);coef=np.linalg.solve(Phi.T@Phi+1e-3*np.eye(deg+1),Phi.T@yo)
pred=np.vander(xs,deg+1,increasing=True)@coef
ax4.plot(xs,truth,"--",lw=1.4,label="Reference");ax4.scatter(xo,yo,s=22,label="Sparse observations",zorder=3);ax4.plot(xs,pred,lw=2,label="Learned approximation")
ax4.set(xlabel=r"$x$",ylabel=r"$u(x)$",title="Data-driven scientific inference");ax4.legend(frameon=False,fontsize=8)
for i,L in enumerate(["(a)","(b)","(c)","(d)"]):
    a=fig.add_subplot(gs[1,i]);a.axis("off");a.text(.5,.5,L,ha="center",va="center",fontsize=13)
fig.savefig("../figures/figure_02_evolution_scientific_discovery.svg",bbox_inches="tight",facecolor="white")
