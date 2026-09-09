"""Reproduce Scientific AI Figure 5 as a single scientific rendering."""
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
rng=np.random.default_rng(42); fig,ax=plt.subplots(figsize=(16,7.2)); ax.set(xlim=(0,16),ylim=(0,7.4)); ax.axis("off")
ink="#30343A"; tissue="#B77B72"; nuclei="#666A78"; scan="#60788B"; mil="#536D73"
x0,y0,w,h=.55,2,2.45,3.25; ax.add_patch(Rectangle((x0,y0),w,h,facecolor="#F4E5DE",edgecolor=ink,lw=.8))
for _ in range(145):
    ax.add_patch(plt.Circle((rng.uniform(x0+.08,x0+w-.08),rng.uniform(y0+.08,y0+h-.08)),rng.uniform(.025,.075),facecolor=nuclei,edgecolor="none",alpha=.72))
for cx,cy,rx,ry in [(1.2,4.35,.42,.25),(2.2,3.75,.5,.3),(1.65,2.75,.45,.28)]:
    th=np.linspace(0,2*np.pi,100); ax.plot(cx+rx*np.cos(th),cy+ry*np.sin(th),color=tissue,lw=2)
sx=3.65; ax.add_patch(Rectangle((sx,2.25),1.6,2.75,facecolor="#F4F5F5",edgecolor=scan,lw=1))
for yy in np.linspace(2.5,4.75,12): ax.plot([sx+.18,sx+1.42],[yy,yy],color=scan,lw=.45,alpha=.7)
wx=5.85; ax.add_patch(Rectangle((wx,2.15),2.25,3.05,facecolor="#F3E2DC",edgecolor=ink,lw=.8))
for _ in range(180): ax.scatter(rng.uniform(wx+.05,wx+2.2),rng.uniform(2.2,5.15),s=rng.uniform(2,9),c=[nuclei],alpha=.45,linewidths=0)
px=8.75; size=.48
for i in range(4):
    for j in range(4): ax.add_patch(Rectangle((px+i*size,3.05+j*size),size-.025,size-.025,facecolor="#ECE1DC",edgecolor=ink,lw=.45))
for i,j in [(0,2),(2,3),(3,1),(1,0)]: ax.add_patch(Rectangle((px+i*size,3.05+j*size),size-.025,size-.025,facecolor="none",edgecolor=mil,lw=1.5))
ax.text(12.1,3.9,r"$z=\sum_i a_i h_i$",fontsize=11); ax.text(12.1,3.45,r"$\sum_i a_i=1$",fontsize=9)
for x,y,t in [(1.78,1.58,"tissue acquisition + staining"),(4.45,1.58,"whole-slide scanning"),(6.97,1.58,"digitized whole-slide image"),(9.69,1.58,"spatially indexed tissue patches"),(12.2,1.58,"slide-level representation"),(14.45,1.58,"patient / slide-level output")]: ax.text(x,y,t,ha="center",va="top",fontsize=8)
for a,b,l in [(3.05,3.62,"optical digitization"),(5.28,5.82,"raster image"),(8.13,8.70,"tile coordinates"),(10.72,11.05,"instances"),(13.25,13.75,"aggregated state")]:
    ax.annotate("",xy=(b,3.65),xytext=(a,3.65),arrowprops=dict(arrowstyle="-|>",lw=.85,color=ink)); ax.text((a+b)/2,3.85,l,ha="center",fontsize=7)
fig.savefig("../figures/figure_05_whole_slide_imaging.svg",bbox_inches="tight",facecolor="white")
