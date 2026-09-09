"""Reproduce Scientific AI Figure 4 as a formal provenance DAG."""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import hashlib
fig,ax=plt.subplots(figsize=(15.5,7.6)); ax.set(xlim=(0,15.5),ylim=(0,8)); ax.axis("off")
ink="#2E3238"; edge="#555B61"; fills={"DATA":"#E9EEF2","ACTIVITY":"#EDE9E3","ENTITY":"#EEF1EC","RELEASE":"#EAE8EF"}
nodes={"raw1":(1,5.9,1.55,.72,"raw/0001","DATA"),"raw2":(1,4.55,1.55,.72,"raw/0002","DATA"),"acq":(3.1,5.25,1.75,.82,"acquire.py","ACTIVITY"),"prov":(3.1,6.85,1.85,.72,"run manifest","ENTITY"),"cur":(5.55,5.25,1.75,.82,"curate.py","ACTIVITY"),"cdata":(7.75,5.25,1.7,.72,"curated/v3","DATA"),"ann":(7.75,6.75,1.7,.72,"annotations/v2","DATA"),"env":(5.55,6.75,1.75,.72,"environment","ENTITY"),"rel":(10.05,5.25,1.85,.82,"release R3","RELEASE"),"doi":(10.05,6.75,1.85,.72,"persistent ID","ENTITY"),"ana":(12.35,5.25,1.75,.82,"analysis.py","ACTIVITY"),"res":(14.45,5.25,1.55,.72,"result/v1","DATA"),"bench":(14.45,3.55,1.55,.72,"benchmark","DATA"),"arch":(10.05,2.25,1.95,.82,"archive copy","RELEASE"),"reuse":(12.35,2.25,1.75,.82,"reuse run","ACTIVITY"),"derived":(14.45,2.25,1.55,.72,"derived/v1","DATA")}
for x,y,w,h,label,typ in nodes.values():
    ax.add_patch(Rectangle((x-w/2,y-h/2),w,h,facecolor=fills[typ],edgecolor=ink,lw=.9,zorder=3)); ax.text(x,y+.08,label,ha="center",va="center",fontsize=8.4); ax.text(x,y-h*.27,typ,ha="center",va="center",fontsize=6.2,color="#666B70")
def ed(a,b,label=None,style="-",rad=0):
    xa,ya,wa,ha,_,_=nodes[a]; xb,yb,wb,hb,_,_=nodes[b]
    ax.add_patch(FancyArrowPatch((xa+np.sign(xb-xa)*wa/2,ya),(xb-np.sign(xb-xa)*wb/2,yb),arrowstyle="-|>",mutation_scale=9,lw=.75,color=edge,linestyle=style,connectionstyle=f"arc3,rad={rad}",zorder=1))
    if label: ax.text((xa+xb)/2,(ya+yb)/2+.12,label,ha="center",fontsize=6.7,color=edge,bbox=dict(facecolor="white",edgecolor="none",pad=.4))
import numpy as np
for a,b,l in [("raw1","acq","used"),("raw2","acq","used"),("prov","acq","qualified"),("acq","cur","generated"),("env","cur","executed-in"),("cur","cdata","generated"),("ann","cdata","annotates"),("cdata","rel","member-of"),("doi","rel","identifies"),("rel","ana","used"),("ana","res","generated"),("res","bench","evaluated-as"),("rel","arch","replicated-to"),("arch","reuse","used"),("reuse","derived","generated"),("bench","reuse","parameterizes")]: ed(a,b,l)
ax.text(10.05,4.55,"sha256:"+hashlib.sha256(b"Scientific AI provenance example release R3").hexdigest()[:12]+"…",ha="center",fontsize=6.6,color="#62676C")
fig.savefig("../figures/figure_04_scientific_data_provenance.svg",bbox_inches="tight",facecolor="white")
