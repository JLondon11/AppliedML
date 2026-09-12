"""Regenerate revised Computer Vision Part II Figure 28.
Scientific source: chapter vehicle motion and radar measurement equations.
No empirical data are synthesized.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arc, FancyArrowPatch, Circle
from matplotlib import transforms
from pathlib import Path

OUT = Path("../figures")
navy="#17324D"; slate="#536B8E"; teal="#2A7F7F"; orange="#C97832"; gray="#6B7280"

fig,axs=plt.subplots(1,2,figsize=(12.8,5.2),constrained_layout=True)

ax=axs[0]; ax.set_aspect("equal"); ax.set_xlim(-.5,8.5); ax.set_ylim(-.5,6.2); ax.axis("off")
ax.annotate("",(8,0),(0,0),arrowprops=dict(arrowstyle="->",lw=1.2,color=gray))
ax.annotate("",(0,5.7),(0,0),arrowprops=dict(arrowstyle="->",lw=1.2,color=gray))
ax.text(8.05,-.15,r"$x$",fontsize=11); ax.text(-.25,5.75,r"$y$",fontsize=11)
p0=np.array([2.0,1.5]); th0=np.deg2rad(24); omega_dt=np.deg2rad(28); th1=th0+omega_dt
mid=th0+omega_dt/2; d=4.0
p1=p0+d*np.array([np.cos(mid),np.sin(mid)])
def vehicle(ax,p,theta,edge):
    w,h=1.6,.8
    rect=Rectangle((p[0]-w/2,p[1]-h/2),w,h,fill=False,lw=1.6,edgecolor=edge)
    rect.set_transform(transforms.Affine2D().rotate_around(p[0],p[1],theta)+ax.transData)
    ax.add_patch(rect); ax.plot(*p,"o",ms=4,color=edge)
vehicle(ax,p0,th0,navy); vehicle(ax,p1,th1,teal)
ax.plot([p0[0],p1[0]],[p0[1],p1[1]],ls="--",lw=1.2,color=gray)
ax.add_patch(FancyArrowPatch(p0,p1,arrowstyle="-|>",mutation_scale=12,lw=1.5,color=teal))
ax.text(*(p0+[-.15,-.55]),r"$(x_{k-1},y_{k-1})$",fontsize=10,ha="center")
ax.text(*(p1+[.25,.55]),r"$(x_k,y_k)$",fontsize=10,ha="center")
for p,t,lab in [(p0,th0,r"$\theta_{k-1}$"),(p1,th1,r"$\theta_k$")]:
    ax.plot([p[0],p[0]+1.25*np.cos(t)],[p[1],p[1]+1.25*np.sin(t)],lw=1.5,color=orange)
    ax.text(p[0]+1.35*np.cos(t),p[1]+1.35*np.sin(t),lab,fontsize=10)
ax.text((p0[0]+p1[0])/2,(p0[1]+p1[1])/2+.35,
        r"$v_{k-1}\Delta t$ along $\beta_{k-1}+\theta_{k-1}+\omega_{k-1}\Delta t/2$",
        fontsize=9,ha="center")
ax.text(.5,-.08,"(a)",transform=ax.transAxes,ha="center",fontsize=10)

ax=axs[1]; ax.set_aspect("equal"); ax.set_xlim(-.5,7.2); ax.set_ylim(-.5,6.2); ax.axis("off")
o=np.array([1.,1.]); target=np.array([5.5,4.5]); vec=target-o; rho=np.linalg.norm(vec); phi=np.arctan2(vec[1],vec[0])
ax.annotate("",(7,1),(1,1),arrowprops=dict(arrowstyle="->",lw=1.2,color=gray))
ax.annotate("",(1,6),(1,1),arrowprops=dict(arrowstyle="->",lw=1.2,color=gray))
ax.text(7.0,.78,r"$p_x$",fontsize=10); ax.text(.72,6.0,r"$p_y$",fontsize=10)
ax.add_patch(Circle(o,.13,fill=False,lw=1.5,edgecolor=navy)); ax.text(.6,.55,"radar",fontsize=9,color=navy)
ax.plot(*target,"o",ms=7,color=teal); ax.text(target[0]+.15,target[1]+.15,"target",fontsize=9)
ax.add_patch(FancyArrowPatch(o,target,arrowstyle="-|>",mutation_scale=12,lw=1.7,color=teal))
ax.text(*(o+.52*vec+[-.1,.25]),r"$\rho=\sqrt{p_x^2+p_y^2}$",fontsize=10)
ax.plot([target[0],target[0]],[1,target[1]],ls=":",lw=1.1,color=gray)
ax.plot([1,target[0]],[target[1],target[1]],ls=":",lw=1.1,color=gray)
ax.text(target[0]-.2,.72,r"$p_x$",fontsize=10); ax.text(.55,target[1]-.05,r"$p_y$",fontsize=10)
ax.add_patch(Arc(o,2.0,2.0,theta1=0,theta2=np.degrees(phi),lw=1.5,color=orange))
ax.text(2.0,1.35,r"$\phi=\tan^{-1}(p_y/p_x)$",fontsize=9,color=orange)
v=np.array([1.2,.45]); ax.add_patch(FancyArrowPatch(target,target+v,arrowstyle="-|>",mutation_scale=12,lw=1.5,color=slate))
ax.text(*(target+v+[.05,.05]),r"$\mathbf{v}=(v_x,v_y)$",fontsize=9)
unit=vec/rho; vr=.95*unit
ax.add_patch(FancyArrowPatch(target,target+vr,arrowstyle="-|>",mutation_scale=12,lw=1.7,color=orange))
ax.text(*(target+vr+[-.05,.2]),r"$\dot{\rho}=\frac{p_xv_x+p_yv_y}{\rho}$",fontsize=9)
ax.text(.5,-.08,"(b)",transform=ax.transAxes,ha="center",fontsize=10)

for ext in ("png","svg","pdf"):
    fig.savefig(OUT/f"Figure_28_Vehicle_State_Transition_and_Radar_Measurement_Geometry.{ext}",
                dpi=400 if ext=="png" else None,bbox_inches="tight",facecolor="white")
