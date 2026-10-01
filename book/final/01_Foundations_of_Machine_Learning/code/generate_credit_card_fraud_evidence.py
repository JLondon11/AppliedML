from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    precision_score, recall_score, f1_score
)

COLORS={"Logistic Regression":"#355C7D","Balanced Logistic":"#4F7C6E","Balanced Random Forest":"#B26E3B"}

def precision_recall_at_budget(y, score, frac=0.01):
    n=max(1,int(np.ceil(len(y)*frac)))
    idx=np.argsort(score)[::-1][:n]
    tp=int(np.sum(y[idx]==1))
    precision=tp/n
    recall=tp/max(1,int(np.sum(y==1)))
    return precision,recall,n

def best_f1_threshold(y, score):
    p,r,t=precision_recall_curve(y,score)
    f=2*p*r/np.maximum(p+r,1e-12)
    i=int(np.nanargmax(f[:-1]))
    return float(t[i]),float(f[i])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="book/final/01_Foundations_of_Machine_Learning/figures")
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)

    ds=fetch_openml(data_id=1597,as_frame=True,parser="auto")
    df=ds.frame.copy()
    target=ds.target.name if hasattr(ds.target,"name") and ds.target.name else "Class"
    if target not in df.columns:
        target="Class"
    df[target]=pd.to_numeric(df[target],errors="coerce").astype(int)
    if "Time" in df.columns:
        df=df.sort_values("Time",kind="mergesort").reset_index(drop=True)
    split=int(len(df)*0.70)
    train=df.iloc[:split].copy(); test=df.iloc[split:].copy()
    Xtr=train.drop(columns=[target]); ytr=train[target].to_numpy()
    Xte=test.drop(columns=[target]); yte=test[target].to_numpy()

    models={
      "Logistic Regression": make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,C=1.0,solver="lbfgs")),
      "Balanced Logistic": make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,C=1.0,solver="lbfgs",class_weight="balanced")),
      "Balanced Random Forest": RandomForestClassifier(n_estimators=120,max_depth=14,min_samples_leaf=2,class_weight="balanced_subsample",n_jobs=-1,random_state=1729)
    }

    rows=[]; curves={}
    for name,model in models.items():
        model.fit(Xtr,ytr)
        score=model.predict_proba(Xte)[:,1]
        apv=average_precision_score(yte,score)
        auc=roc_auc_score(yte,score)
        thr,bf=best_f1_threshold(yte,score)
        pred=(score>=thr).astype(int)
        p1,r1,n1=precision_recall_at_budget(yte,score,.01)
        rows.append(dict(
            model=name,average_precision=apv,roc_auc=auc,best_f1=bf,
            best_f1_threshold=thr,precision_at_top_1pct=p1,recall_at_top_1pct=r1,
            top_1pct_review_count=n1,
            threshold_precision=precision_score(yte,pred,zero_division=0),
            threshold_recall=recall_score(yte,pred,zero_division=0),
            threshold_f1=f1_score(yte,pred,zero_division=0)
        ))
        curves[name]=precision_recall_curve(yte,score)[:2]

    res=pd.DataFrame(rows)
    res.to_csv(out/"ch01_fraud_benchmark_table.csv",index=False)
    meta={
      "dataset":"OpenML creditcard, data_id=1597 (ULB/Worldline benchmark)",
      "rows":int(len(df)),"fraud_cases":int(df[target].sum()),
      "train_rows":int(len(train)),"test_rows":int(len(test)),
      "split":"chronological earliest 70% train, latest 30% test after sorting by Time",
      "target":target
    }
    (out/"ch01_fraud_provenance.json").write_text(json.dumps(meta,indent=2))

    fig,axs=plt.subplots(1,2,figsize=(10.4,4.15))
    for name,(p,r) in curves.items():
        axs[0].plot(r,p,lw=2,label=name,color=COLORS[name])
    prevalence=float(yte.mean())
    axs[0].axhline(prevalence,ls="--",lw=1,color="#777777",label=f"Test prevalence ({prevalence:.3%})")
    axs[0].set_xlabel("Recall")
    axs[0].set_ylabel("Precision")
    axs[0].set_xlim(0,1); axs[0].set_ylim(0,1)
    axs[0].legend(frameon=False,fontsize=7)

    order=list(models)
    x=np.arange(len(order)); w=.34
    pvals=[res.set_index("model").loc[n,"precision_at_top_1pct"] for n in order]
    rvals=[res.set_index("model").loc[n,"recall_at_top_1pct"] for n in order]
    axs[1].bar(x-w/2,pvals,width=w,label="Precision",color="#355C7D")
    axs[1].bar(x+w/2,rvals,width=w,label="Recall",color="#B26E3B")
    axs[1].set_xticks(x,["LR","Balanced LR","Balanced RF"])
    axs[1].set_ylabel("Top-1% review metric")
    axs[1].set_ylim(0,1)
    axs[1].legend(frameon=False,fontsize=8)
    for i,ax in enumerate(axs):
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(direction="out")
        ax.text(.5,-.23,f"({chr(97+i)})",transform=ax.transAxes,ha="center",va="top",fontsize=10)
    fig.tight_layout(w_pad=2.0)
    for ext in ("png","svg","pdf"):
        fig.savefig(out/f"ch01_credit_card_fraud_imbalance.{ext}",dpi=300 if ext=="png" else None,bbox_inches="tight")
    plt.close(fig)

if __name__=="__main__":
    main()
