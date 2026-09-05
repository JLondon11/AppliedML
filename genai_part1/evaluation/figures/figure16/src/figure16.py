from pathlib import Path
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
HERE=Path(__file__).resolve().parent.parent
df=pd.read_csv(HERE/"data"/"figure16_cifar10_published.csv"); OUT=HERE/"outputs"; OUT.mkdir(parents=True,exist_ok=True)
perf=pd.DataFrame(index=df.index)
perf["FID"]=1-(df.FID-df.FID.min())/(df.FID.max()-df.FID.min())
for c in ["Precision","Recall","Coverage"]: perf[c]=(df[c]-df[c].min())/(df[c].max()-df[c].min())
cmap=LinearSegmentedColormap.from_list("premium_blue",["#F7F9FB","#D8E3EA","#8AA6B6","#3E6478","#173A52"])
fig,ax=plt.subplots(figsize=(7.4,4.7),dpi=240)
im=ax.imshow(perf[["FID","Precision","Recall","Coverage"]].to_numpy(),cmap=cmap,vmin=0,vmax=1,aspect="auto")
ax.set_xticks(range(4),["FID ↓","Precision ↑","Recall ↑","Coverage ↑"]); ax.set_yticks(range(len(df)),df.Model)
for i,row in df.iterrows():
    for j,val in enumerate([row.FID,row.Precision,row.Recall,row.Coverage]):
        ax.text(j,i,f"{val:.2f}",ha="center",va="center",color="white" if perf.iloc[i,j]>.62 else "#17212A",fontsize=9)
ax.set_xticks(np.arange(-.5,4,1),minor=True); ax.set_yticks(np.arange(-.5,len(df),1),minor=True)
ax.grid(which="minor",linewidth=.45,alpha=.35); ax.tick_params(which="minor",bottom=False,left=False)
cb=fig.colorbar(im,ax=ax,fraction=.035,pad=.025); cb.set_label("Within-column relative performance")
cb.set_ticks([0,.5,1]); cb.ax.set_yticklabels(["lower","mid","higher"])
fig.tight_layout(); fig.savefig(OUT/"figure16.png",bbox_inches="tight"); fig.savefig(OUT/"figure16.svg",bbox_inches="tight"); plt.close(fig)
