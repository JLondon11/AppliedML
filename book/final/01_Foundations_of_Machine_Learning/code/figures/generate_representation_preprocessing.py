"""Generate Figure 1.8: quantitative effects of representation preprocessing."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PALETTE=["#355C7D","#4F7C6E","#B26E3B","#7A6AA6"]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="book/final/01_Foundations_of_Machine_Learning/figures")
    args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    X,y=load_breast_cancer(return_X_y=True)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.30,stratify=y,random_state=42)
    scaler=StandardScaler().fit(Xtr)
    Xs=scaler.transform(Xtr); Xte_s=scaler.transform(Xte)
    pca=PCA().fit(Xs)
    raw_scale=np.std(Xtr,axis=0); std_scale=np.std(Xs,axis=0)
    raw_gram=(Xtr.T@Xtr)/len(Xtr)+1e-8*np.eye(Xtr.shape[1])
    std_gram=(Xs.T@Xs)/len(Xs)+1e-8*np.eye(Xs.shape[1])
    cond_raw=float(np.linalg.cond(raw_gram)); cond_std=float(np.linalg.cond(std_gram))
    Xtr_p=pca.transform(Xs)[:,:10]; Xte_p=pca.transform(Xte_s)[:,:10]

    configs=[
      ("raw",Xtr,Xte),
      ("standardized",Xs,Xte_s),
      ("PCA-10",Xtr_p,Xte_p)
    ]
    aucs=[]
    for name,a,b in configs:
        m=LogisticRegression(max_iter=1500).fit(a,ytr)
        aucs.append(roc_auc_score(yte,m.predict_proba(b)[:,1]))

    fig,axs=plt.subplots(1,3,figsize=(10.2,3.15))
    order=np.argsort(raw_scale)
    axs[0].semilogy(np.arange(len(order)),raw_scale[order],marker="o",ms=3,lw=1.2,label="raw",color=PALETTE[2])
    axs[0].semilogy(np.arange(len(order)),std_scale[order],marker="o",ms=3,lw=1.2,label="standardized",color=PALETTE[0])
    axs[0].set_xlabel("Feature index (sorted by raw scale)")
    axs[0].set_ylabel("Training-set standard deviation")
    axs[0].legend(frameon=False,fontsize=8)

    axs[1].bar(["raw","standardized"],[cond_raw,cond_std],color=[PALETTE[2],PALETTE[1]])
    axs[1].set_yscale("log")
    axs[1].set_ylabel("Condition number of regularized Gram matrix")

    cum=np.cumsum(pca.explained_variance_ratio_)
    axs[2].bar(["raw","standardized","PCA-10"],aucs,
               color=[PALETTE[0],PALETTE[1],PALETTE[2]],width=.62)
    axs[2].set_ylim(.90,1.0)
    axs[2].set_ylabel("Held-out ROC-AUC")
    axs[2].tick_params(axis="x",rotation=15)
    axs[2].text(.03,.08,f"PCA-10 cumulative variance = {cum[9]:.3f}",
                transform=axs[2].transAxes,fontsize=8)

    for i,ax in enumerate(axs):
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(direction="out")
        ax.text(.5,-.23,f"({chr(97+i)})",transform=ax.transAxes,ha="center",va="top",fontsize=10)
    fig.tight_layout(w_pad=2.0)
    for ext in ("png","svg","pdf"):
        fig.savefig(out/f"figure_01_08_representation_preprocessing.{ext}",
                    dpi=300 if ext=="png" else None,bbox_inches="tight",pad_inches=.03)
    plt.close(fig)

if __name__=="__main__":
    main()
