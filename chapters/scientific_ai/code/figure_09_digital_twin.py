"""Reproduce Scientific AI Figure 9 as a formal state-space/data-assimilation graph."""
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
fig,ax=plt.subplots(figsize=(14.5,7.2)); ax.set(xlim=(0,14.5),ylim=(0,7.2)); ax.axis("off")
ink="#30343A"; state="#536A78"; obs="#687D78"; control="#846B59"; unc="#77727F"; xs=[2.2,5.3,8.4,11.5]
for i,x in enumerate(xs):
    ax.add_patch(Circle((x,4.35),.37,facecolor="white",edgecolor=state,lw=1.25)); ax.text(x,4.35,rf"$x_{i}$",ha="center",va="center",fontsize=12)
    ax.add_patch(Circle((x,6),.32,facecolor="white",edgecolor=obs,lw=1.15)); ax.text(x,6,rf"$y_{i}$",ha="center",va="center",fontsize=11)
    ax.add_patch(FancyArrowPatch((x,5.66),(x,4.74),arrowstyle="-|>",mutation_scale=10,lw=.9,color=obs))
for i in range(3):
    ax.add_patch(FancyArrowPatch((xs[i]+.39,4.35),(xs[i+1]-.39,4.35),arrowstyle="-|>",mutation_scale=11,lw=1,color=ink)); ax.text((xs[i]+xs[i+1])/2,4.62,rf"$\mathcal{{M}}_{{{i}\to{i+1}}}$",ha="center")
    ax.add_patch(Circle((xs[i],2.55),.31,facecolor="white",edgecolor=control,lw=1.15)); ax.text(xs[i],2.55,rf"$u_{i}$",ha="center",va="center",fontsize=11)
for x in [8.4,11.5]:
    yy=np.linspace(-1,1,120); dens=np.exp(-.5*(yy/.38)**2); scale=.42 if x==8.4 else .62
    ax.plot(x+scale*dens,4.35+.72*yy,color=unc,lw=.8); ax.plot(x-scale*dens,4.35+.72*yy,color=unc,lw=.8)
ax.text(7.25,.78,r"$x_{k+1}=\mathcal{M}_k(x_k,u_k)+\eta_k$"+"
"+r"$y_k=\mathcal{H}_k(x_k)+\epsilon_k$",ha="center",fontsize=10.2)
fig.savefig("../figures/figure_09_scientific_digital_twin.svg",bbox_inches="tight",facecolor="white")
