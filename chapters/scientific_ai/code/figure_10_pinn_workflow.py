"""Reproduce Scientific AI Figure 10 PINN workflow from mathematical operators."""
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
rng=np.random.default_rng(10); fig,ax=plt.subplots(figsize=(14.8,7.2)); ax.set(xlim=(0,15),ylim=(0,7.4)); ax.axis("off")
ink="#30343A"; navy="#526A7A"; teal="#607B76"; umber="#8A6A54"; slate="#707984"; violet="#756F80"; pale="#F3F4F4"
x0,y0,w,h=.75,1.35,3.15,4.75; ax.add_patch(Rectangle((x0,y0),w,h,facecolor="white",edgecolor=ink,lw=.85))
ax.scatter(rng.uniform(x0+.15,x0+w-.15,95),rng.uniform(y0+.15,y0+h-.15,95),s=8,facecolors="none",edgecolors=slate,linewidths=.45)
yb=np.linspace(y0+.12,y0+h-.12,18); ax.scatter(np.full_like(yb,x0),yb,s=15,marker="s",color=umber); ax.scatter(np.full_like(yb,x0+w),yb,s=15,marker="s",color=umber)
xi=np.linspace(x0+.08,x0+w-.08,22); ax.scatter(xi,np.full_like(xi,y0),s=15,marker="^",color=teal)
ax.scatter(rng.uniform(x0+.25,x0+w-.25,14),rng.uniform(y0+.35,y0+h-.25,14),s=22,marker="o",facecolor=navy,edgecolor="white",linewidth=.35)
for x,y,w,h,text,edge in [(5.15,3.25,2,1.2,r"$u_\theta(x,t)$",navy),(8.15,3.25,2.15,1.2,r"$\partial_tu_\theta,\nabla u_\theta,\nabla^2u_\theta$",violet),(11.25,4.45,2.65,1.05,r"$r_\theta=\mathcal{N}[u_\theta]-f$",umber)]:
    ax.add_patch(Rectangle((x,y),w,h,facecolor=pale if x==5.15 else "white",edgecolor=edge,lw=1)); ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=11)
ax.add_patch(FancyArrowPatch((3.9,3.85),(5.07,3.85),arrowstyle="-|>",mutation_scale=10,lw=.9,color=ink)); ax.add_patch(FancyArrowPatch((7.23,3.85),(8.07,3.85),arrowstyle="-|>",mutation_scale=10,lw=.9,color=ink))
for y,sym,desc,c in [(5.95,r"$\mathcal{L}_{obs}$","observational mismatch",navy),(4.25,r"$\mathcal{L}_{phys}$",r"$\|r_\theta\|_2^2$",umber),(2.55,r"$\mathcal{L}_{BC}$","boundary mismatch",slate),(1.15,r"$\mathcal{L}_{IC}$","initial mismatch",teal)]:
    ax.text(12.05,y,sym,fontsize=10.5,color=c); ax.text(13,y,desc,fontsize=7.6)
ax.text(7.95,.70,r"$\mathcal{L}(\theta)=\lambda_{obs}\mathcal{L}_{obs}+\lambda_{phys}\mathcal{L}_{phys}+\lambda_{BC}\mathcal{L}_{BC}+\lambda_{IC}\mathcal{L}_{IC}$",ha="center",fontsize=11.2)
fig.savefig("../figures/figure_10_physics_informed_workflow.svg",bbox_inches="tight",facecolor="white")
