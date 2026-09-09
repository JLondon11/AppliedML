"""Reproduce Scientific AI Figure 12 with formal retinal diagnostic operators."""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
fig,ax=plt.subplots(figsize=(15.6,7)); ax.set(xlim=(0,15.6),ylim=(0,7)); ax.axis("off")
ink="#30343A"; fundus="#6E665E"; octc="#566D7A"; qc="#73777C"; seg="#6A7B73"; feat="#6D6877"; cal="#846A56"; pale="#F4F5F5"; line="#747A7F"
def node(x,y,w,h,main,sub="",edge=ink,fill="white"):
    ax.add_patch(Rectangle((x-w/2,y-h/2),w,h,facecolor=fill,edgecolor=edge,lw=.95,zorder=3)); ax.text(x,y+.1 if sub else y,main,ha="center",va="center",fontsize=9.5)
    if sub: ax.text(x,y-h*.27,sub,ha="center",va="center",fontsize=7,color="#666B70")
def arr(a,b,c,d):
    ax.add_patch(FancyArrowPatch((a,b),(c,d),arrowstyle="-|>",mutation_scale=9,lw=.8,color=line,zorder=1))
node(1.45,5,2.15,.8,"fundus image",r"$I_F(x,y)$",fundus,pale); node(1.45,2.25,2.15,.8,"OCT volume",r"$I_O(x,y,z)$",octc,pale)
node(4.2,5,2.15,.82,r"$Q_F(I_F)$","image-quality assessment",qc); node(4.2,2.25,2.15,.82,r"$Q_O(I_O)$","image-quality assessment",qc)
node(7.15,5,2.2,.9,r"$S_F(I_F)$","vessels / lesions / disc",seg); node(7.15,2.25,2.2,.9,r"$S_O(I_O)$","retinal layers / lesions",seg)
node(10,5,2.15,.9,r"$E_F(I_F,S_F)$","fundus representation",feat,pale); node(10,2.25,2.15,.9,r"$E_O(I_O,S_O)$","OCT representation",feat,pale)
node(12.45,3.62,2.1,.95,r"$z=\mathcal{F}(h_F,h_O)$","retinal representation",ink); node(14.35,3.62,1.65,.95,r"$\hat p_c$","calibrated risk",cal,pale)
for a,b,c,d in [(2.55,5,3.1,5),(2.55,2.25,3.1,2.25),(5.95,5,6.02,5),(5.95,2.25,6.02,2.25),(8.27,5,8.92,5),(8.27,2.25,8.92,2.25),(11.1,5,11.42,4.02),(11.1,2.25,11.42,3.22),(13.52,3.62,13.5,3.62)]: arr(a,b,c,d)
ax.text(14.35,2.48,r"$d=\mathbf{1}[\hat p_c\geq\tau_c]$",ha="center",fontsize=10); ax.text(14.35,1.98,"referral decision support",ha="center",fontsize=7.7)
ax.text(11.8,.72,r"$\hat p_c=\mathcal{C}(g(z))$ with calibration evaluated independently of discrimination",ha="center",fontsize=9)
fig.savefig("../figures/figure_12_retinal_imaging_pipeline.svg",bbox_inches="tight",facecolor="white")
