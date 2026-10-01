from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

def quantize(w,bits):
    qmax=2**(bits-1)-1
    scale=max(np.max(np.abs(w)),1e-12)/qmax
    q=np.clip(np.round(w/scale),-qmax,qmax).astype(np.int16)
    return q,scale

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="book/final/01_Foundations_of_Machine_Learning/figures")
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)

    d=load_breast_cancer()
    Xtr,Xte,ytr,yte=train_test_split(d.data,d.target,test_size=.30,stratify=d.target,random_state=1729)
    sc=StandardScaler().fit(Xtr)
    Xtr=sc.transform(Xtr); Xte=sc.transform(Xte)
    m=LogisticRegression(max_iter=2000,C=1.0).fit(Xtr,ytr)
    w=m.coef_.ravel(); b=float(m.intercept_[0])

    rows=[]
    p=1/(1+np.exp(-(Xte@w+b)))
    rows.append({"precision":"float64","bits_per_weight":64,"storage_bytes":int(w.size*8),
                 "accuracy":accuracy_score(yte,p>=.5),"roc_auc":roc_auc_score(yte,p),
                 "log_loss":log_loss(yte,p)})
    for bits in (8,4,3,2):
        q,s=quantize(w,bits)
        wq=q.astype(float)*s
        pq=1/(1+np.exp(-(Xte@wq+b)))
        theoretical=int(np.ceil(w.size*bits/8))
        rows.append({"precision":f"int{bits}","bits_per_weight":bits,"storage_bytes":theoretical,
                     "accuracy":accuracy_score(yte,pq>=.5),"roc_auc":roc_auc_score(yte,pq),
                     "log_loss":log_loss(yte,pq)})
    df=pd.DataFrame(rows)
    df.to_csv(out/"ch01_edge_quantization_benchmark.csv",index=False)

    order=df["precision"].tolist()
    fig,axs=plt.subplots(1,2,figsize=(10.0,3.85))
    axs[0].bar(order,df["storage_bytes"]/1024,color=["#243447","#4F7C6E","#B26E3B","#7A6AA6","#657A8A"])
    axs[0].set_ylabel("Coefficient storage (KiB)")
    axs[0].set_xlabel("Weight representation")
    axs[1].plot(order,df["accuracy"],marker="o",lw=1.8,color="#355C7D",label="Accuracy")
    axs[1].plot(order,df["roc_auc"],marker="s",lw=1.8,color="#B26E3B",label="ROC--AUC")
    axs[1].set_ylim(.85,1.01)
    axs[1].set_ylabel("Held-out metric")
    axs[1].set_xlabel("Weight representation")
    axs[1].legend(frameon=False,fontsize=8)
    for i,ax in enumerate(axs):
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(direction="out")
        ax.text(.5,-.22,f"({chr(97+i)})",transform=ax.transAxes,ha="center",va="top",fontsize=10)
    fig.tight_layout(w_pad=2.0,pad=.25)
    for ext in ("png","svg","pdf"):
        fig.savefig(out/f"ch01_edge_quantization_tradeoff.{ext}",dpi=300 if ext=="png" else None,bbox_inches="tight",pad_inches=.02)
    plt.close(fig)

    meta={
      "dataset":"scikit-learn Wisconsin Diagnostic Breast Cancer",
      "split":"stratified 70/30, random_state=1729",
      "model":"standardized logistic regression",
      "quantization":"symmetric per-tensor weight-only quantization; theoretical packed coefficient storage"
    }
    (out/"ch01_edge_quantization_provenance.json").write_text(json.dumps(meta,indent=2))

if __name__=="__main__":
    main()
