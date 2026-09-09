"""Reproduce Scientific AI Figure 3 with one coherent diffusion system."""
import numpy as np, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
np.random.seed(21)
x=np.linspace(0,1,320); times=np.linspace(0,.22,180); nu=.035
u0=np.exp(-((x-.32)/.105)**2)-.62*np.exp(-((x-.72)/.13)**2)
modes=np.arange(1,31); B=np.sin(np.pi*np.outer(x,modes))
coef=2*np.trapezoid(u0[:,None]*B,x,axis=0)
field=lambda tt: B@(coef*np.exp(-nu*(np.pi*modes)**2*tt))
U=np.vstack([field(tt) for tt in times])
fig=plt.figure(figsize=(15.6,6.15)); gs=GridSpec(2,5,figure=fig,height_ratios=[1,.055],wspace=.42,hspace=.10)
tobs=.11; truth=field(tobs); idx=np.linspace(12,307,22,dtype=int); noise=.025
obs=truth[idx]+np.random.normal(0,noise,len(idx))
ax=fig.add_subplot(gs[0,0]); ax.plot(x,truth,lw=1.55,label="physical state"); ax.errorbar(x[idx],obs,yerr=noise,fmt="o",ms=2.8,lw=.6,capsize=1.5,label="measurements")
ax.set(xlabel=r"$x$",ylabel=r"$u(x,t^*)$",title="Measurement"); ax.legend(frameon=False,fontsize=7)
ax=fig.add_subplot(gs[0,1]); im=ax.imshow(U,origin="lower",aspect="auto",extent=[0,1,times[0],times[-1]],cmap="coolwarm",vmin=-1,vmax=1)
ax.axhline(tobs,ls="--",lw=.7); ax.set(xlabel=r"$x$",ylabel=r"$t$",title="Mechanistic simulation"); fig.colorbar(im,ax=ax,fraction=.046,pad=.025,label=r"$u(x,t)$")
ax=fig.add_subplot(gs[0,2]); centers=np.linspace(0,1,28)
rbf=lambda a,c,l=.09: np.exp(-((a[:,None]-c[None,:])**2)/(2*l*l))
Phi=rbf(x[idx],centers); w=np.linalg.solve(Phi.T@Phi+2e-3*np.eye(len(centers)),Phi.T@obs); pred=rbf(x,centers)@w
ax.plot(x,truth,"--",lw=1.3,label="reference"); ax.plot(x,pred,lw=1.7,label="learned field"); ax.scatter(x[idx],obs,s=10,zorder=3,label="measurements")
ax.set(xlabel=r"$x$",ylabel=r"$u(x,t^*)$",title="Scientific ML reconstruction"); ax.legend(frameon=False,fontsize=7)
ax=fig.add_subplot(gs[0,3]); Uc=U-U.mean(axis=0,keepdims=True); _,S,Vt=np.linalg.svd(Uc,full_matrices=False); scores=Uc@Vt[:3].T
sc=ax.scatter(scores[:,0],scores[:,1],c=times,s=11,cmap="viridis",linewidths=0); ax.plot(scores[:,0],scores[:,1],lw=.55,alpha=.55)
ax.set(xlabel=r"$z_1$",ylabel=r"$z_2$",title="Shared latent representation"); fig.colorbar(sc,ax=ax,fraction=.046,pad=.025,label=r"$t$")
ax=fig.add_subplot(gs[0,4]); candidate=np.linspace(.02,.98,120)
kernel=lambda a,b,l=.12: np.exp(-.5*((a[:,None]-b[None,:])/l)**2); selected=[.08,.92]
for _ in range(6):
    Xs=np.array(selected); K=kernel(Xs,Xs)+1e-5*np.eye(len(Xs)); Ks=kernel(candidate,Xs)
    var=1-np.sum((np.linalg.solve(K,Ks.T).T)*Ks,axis=1); selected.append(float(candidate[np.argmax(var)]))
Xs=np.array(selected); K=kernel(Xs,Xs)+1e-5*np.eye(len(Xs)); Ks=kernel(candidate,Xs); var=np.maximum(1-np.sum((np.linalg.solve(K,Ks.T).T)*Ks,axis=1),0)
ax.plot(candidate,var,lw=1.5,label="posterior variance"); ax.scatter(Xs,np.zeros_like(Xs),s=17,zorder=3,label="selected experiments")
ax.set(xlabel=r"candidate location $x$",ylabel=r"$\mathrm{Var}[u(x)]$",title="Autonomous experiment selection"); ax.legend(frameon=False,fontsize=7)
for i,L in enumerate(["(a)","(b)","(c)","(d)","(e)"]):
    a=fig.add_subplot(gs[1,i]);a.axis("off");a.text(.5,.55,L,ha="center",va="center")
fig.savefig("../figures/figure_03_evolution_scientific_ai.svg",bbox_inches="tight",facecolor="white")
