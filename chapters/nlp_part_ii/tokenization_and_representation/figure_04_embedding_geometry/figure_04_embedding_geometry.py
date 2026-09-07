from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
plt.rcParams["svg.fonttype"]="none"
HERE=Path(__file__).resolve().parent
df=pd.read_csv(HERE/"figure_04_glove_subset_and_pca.csv")
dims=[c for c in df if c.startswith("d")]
raw={r.word:r[dims].to_numpy(float) for _,r in df.iterrows()}
groups={"animals":["dog","cat","wolf"],"vehicles":["car","truck","bus","train","plane"],"finance":["loan","money","credit","investment"],"geography":["river","shore","water","stream"]}
words=list(raw); X=np.array([raw[w] for w in words]); pca=PCA(n_components=2); Z=pca.fit_transform(X)
coord={w:Z[i] for i,w in enumerate(words)}
fb=pca.transform([np.mean([raw[w] for w in ["bank","loan","money","credit","investment"]],axis=0)])[0]
gb=pca.transform([np.mean([raw[w] for w in ["bank","river","shore","water","stream"]],axis=0)])[0]
fig,axs=plt.subplots(1,3,figsize=(13.5,4.6))
ax=axs[0]
for label,g in groups.items():
    xy=np.array([coord[w] for w in g]); ax.scatter(xy[:,0],xy[:,1],s=34,label=label)
    for w in g: ax.annotate(w,coord[w],xytext=(4,4),textcoords="offset points",fontsize=8)
ax.set(xlabel="PC 1",ylabel="PC 2"); ax.legend(frameon=False,fontsize=8); ax.text(.5,-.2,"(a)",transform=ax.transAxes,ha="center",va="top")
ax=axs[1]
for g,label in [(groups["finance"],"financial context"),(groups["geography"],"geographic context")]:
    xy=np.array([coord[w] for w in g]); ax.scatter(xy[:,0],xy[:,1],s=30,label=label)
    for w in g: ax.annotate(w,coord[w],xytext=(4,4),textcoords="offset points",fontsize=8)
b=coord["bank"]; ax.scatter([b[0]],[b[1]],s=45,marker="x"); ax.annotate("bank",b,xytext=(5,5),textcoords="offset points",fontsize=8)
ax.scatter([fb[0],gb[0]],[fb[1],gb[1]],s=52)
ax.annotate("bank | finance",fb,xytext=(5,5),textcoords="offset points",fontsize=8); ax.annotate("bank | geography",gb,xytext=(5,5),textcoords="offset points",fontsize=8)
ax.annotate("",xy=fb,xytext=b,arrowprops={"arrowstyle":"->","linewidth":1}); ax.annotate("",xy=gb,xytext=b,arrowprops={"arrowstyle":"->","linewidth":1})
ax.set(xlabel="PC 1",ylabel="PC 2"); ax.legend(frameon=False,fontsize=8); ax.text(.5,-.2,"(b)",transform=ax.transAxes,ha="center",va="top")
ax=axs[2]; cent={}
for label,g in groups.items():
    c=pca.transform([np.mean([raw[w] for w in g],axis=0)])[0]; cent[label]=c; ax.scatter([c[0]],[c[1]],s=58); ax.annotate(label,c,xytext=(5,5),textcoords="offset points",fontsize=9)
ax.scatter([b[0]],[b[1]],s=38,marker="x"); ax.annotate("bank",b,xytext=(5,5),textcoords="offset points",fontsize=8)
for t in [cent["finance"],cent["geography"]]: ax.annotate("",xy=t,xytext=b,arrowprops={"arrowstyle":"->","linewidth":1.1})
ax.annotate("",xy=cent["vehicles"],xytext=cent["animals"],arrowprops={"arrowstyle":"->","linewidth":1.1})
ax.set(xlabel="PC 1",ylabel="PC 2"); ax.text(.5,-.2,"(c)",transform=ax.transAxes,ha="center",va="top")
for ax in axs: ax.tick_params(direction="out"); ax.margins(.12)
fig.tight_layout(w_pad=2)
fig.savefig(HERE/"figure_04_embedding_geometry.svg",bbox_inches="tight")
fig.savefig(HERE/"figure_04_embedding_geometry.png",dpi=300,bbox_inches="tight")
