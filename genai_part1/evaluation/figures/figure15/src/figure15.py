from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.spatial import cKDTree
SEED=15015; N=5000; B=250
rng=np.random.default_rng(SEED); OUT=Path(__file__).resolve().parent.parent/"outputs"; OUT.mkdir(parents=True,exist_ok=True)
centers=np.array([[-2.,-2.],[-2.,2.],[2.,-2.],[2.,2.]])
def sample(c,s,n=N):
    idx=rng.integers(0,len(c),n); return c[idx]+rng.normal(0,s,size=(n,2))
real=sample(centers,.48)
conds={"Mode-seeking":sample(centers[:2],.30),"Balanced":sample(centers,.50),"Over-dispersed":sample(centers,.82),"Shifted":sample(centers+np.array([.45,-.25]),.48)}
rt=cKDTree(real); d,_=rt.query(real,k=2); tau=np.quantile(d[:,1],.95)
def pr(g):
    gt=cKDTree(g); dg,_=rt.query(g,k=1); dr,_=gt.query(real,k=1)
    return np.mean(dg<=tau),np.mean(dr<=tau)
rows=[]
for name,g in conds.items():
    p,r=pr(g); boots=np.array([pr(g[rng.integers(0,len(g),len(g))]) for _ in range(B)])
    rows.append([name,p,r,boots[:,0].std(ddof=1),boots[:,1].std(ddof=1)])
df=pd.DataFrame(rows,columns=["model_condition","precision","recall","precision_bootstrap_sd","recall_bootstrap_sd"])
df.to_csv(OUT/"figure15_results.csv",index=False)
plt.rcParams.update({"font.family":"serif","font.size":10,"axes.labelsize":11,"legend.fontsize":9})
fig,ax=plt.subplots(figsize=(7.1,5.0),dpi=220)
for (_,r),m in zip(df.iterrows(),["o","s","^","D"]):
    ax.errorbar(r.recall,r.precision,xerr=r.recall_bootstrap_sd,yerr=r.precision_bootstrap_sd,fmt=m,ms=7,capsize=3,linewidth=1.1,label=r.model_condition)
ax.set(xlabel="Empirical support recall",ylabel="Empirical support precision",xlim=(0,1.03),ylim=(0,1.03))
ax.grid(True,linewidth=.5,alpha=.28); ax.legend(frameon=False,loc="lower left"); fig.tight_layout()
fig.savefig(OUT/"figure15.png",bbox_inches="tight"); fig.savefig(OUT/"figure15.svg",bbox_inches="tight"); plt.close(fig)
