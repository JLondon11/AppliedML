from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, f1_score, precision_score, recall_score

COLORS={"Logistic Regression":"#355C7D","Balanced Logistic":"#4F7C6E","Random Forest":"#B26E3B"}

def best_threshold(y, s):
    p,r,t=precision_recall_curve(y,s)
    f=2*p*r/np.maximum(p+r,1e-12)
    i=int(np.nanargmax(f[:-1]))
    return float(t[i]),float(f[i]),float(p[i]),float(r[i])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="book/final/01_Foundations_of_Machine_Learning/figures")
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)

    d=fetch_ucirepo(id=601)
    X=d.data.features.copy()
    y=d.data.targets["Machine failure"].astype(int).to_numpy()

    # Drop identifiers if present; retain machine/process variables only.
    X=X.drop(columns=[c for c in ["UID","Product ID"] if c in X.columns],errors="ignore")
    cat=[c for c in X.columns if str(X[c].dtype)=="object"]
    num=[c for c in X.columns if c not in cat]

    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.30,stratify=y,random_state=1729)

    pre=ColumnTransformer([
      ("num",StandardScaler(),num),
      ("cat",OneHotEncoder(handle_unknown="ignore"),cat)
    ])

    models={
      "Logistic Regression": make_pipeline(pre,LogisticRegression(max_iter=1500)),
      "Balanced Logistic": make_pipeline(pre,LogisticRegression(max_iter=1500,class_weight="balanced")),
      "Random Forest": make_pipeline(
          ColumnTransformer([("num","passthrough",num),("cat",OneHotEncoder(handle_unknown="ignore"),cat)]),
          RandomForestClassifier(n_estimators=400,max_depth=10,min_samples_leaf=2,class_weight="balanced_subsample",random_state=1729,n_jobs=-1)
      )
    }

    rows=[]; curves={}
    for name,m in models.items():
        m.fit(Xtr,ytr)
        s=m.predict_proba(Xte)[:,1]
        th,bf,bp,br=best_threshold(yte,s)
        pred=(s>=th).astype(int)
        rows.append({
          "model":name,
          "average_precision":average_precision_score(yte,s),
          "roc_auc":roc_auc_score(yte,s),
          "best_f1":bf,
          "best_threshold":th,
          "precision":precision_score(yte,pred,zero_division=0),
          "recall":recall_score(yte,pred,zero_division=0),
          "failure_prevalence_test":float(np.mean(yte))
        })
        p,r,_=precision_recall_curve(yte,s)
        curves[name]=(p,r)

    res=pd.DataFrame(rows)
    res.to_csv(out/"ch01_predictive_maintenance_benchmark_table.csv",index=False)

    # Feature-space diagnostic: torque x rotational speed, colored by failure label.
    # Use a deterministic subsample for readability.
    idx=np.arange(len(X))
    rng=np.random.default_rng(1729)
    normal=idx[y==0]; fail=idx[y==1]
    n_normal=min(1800,len(normal))
    show=np.r_[rng.choice(normal,n_normal,replace=False),fail]
    showX=X.iloc[show]
    showy=y[show]

    fig,axs=plt.subplots(1,2,figsize=(10.2,4.05))
    axs[0].scatter(showX.loc[showy==0,"Rotational speed"].astype(float),
                   showX.loc[showy==0,"Torque"].astype(float),
                   s=8,alpha=.22,color="#657A8A",label="Normal")
    axs[0].scatter(showX.loc[showy==1,"Rotational speed"].astype(float),
                   showX.loc[showy==1,"Torque"].astype(float),
                   s=18,alpha=.75,color="#B26E3B",label="Failure")
    axs[0].set_xlabel("Rotational speed (rpm)")
    axs[0].set_ylabel("Torque (Nm)")
    axs[0].legend(frameon=False,fontsize=8)

    for name,(p,r) in curves.items():
        axs[1].plot(r,p,lw=2,label=name,color=COLORS[name])
    prev=float(np.mean(yte))
    axs[1].axhline(prev,ls="--",lw=1,color="#777777",label=f"Test prevalence ({prev:.2%})")
    axs[1].set_xlabel("Recall")
    axs[1].set_ylabel("Precision")
    axs[1].set_xlim(0,1); axs[1].set_ylim(0,1)
    axs[1].legend(frameon=False,fontsize=7)

    for i,ax in enumerate(axs):
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(direction="out")
        ax.text(.5,-.22,f"({chr(97+i)})",transform=ax.transAxes,ha="center",va="top",fontsize=10)
    fig.tight_layout(w_pad=2.0,pad=.3)
    for ext in ("png","svg","pdf"):
        fig.savefig(out/f"ch01_predictive_maintenance_ai4i.{ext}",dpi=300 if ext=="png" else None,bbox_inches="tight",pad_inches=.02)
    plt.close(fig)

    meta={
      "dataset":"UCI AI4I 2020 Predictive Maintenance Dataset, id=601",
      "instances":int(len(X)),
      "failure_cases":int(y.sum()),
      "split":"stratified 70/30 holdout, random_state=1729",
      "note":"Dataset is synthetic but designed to reflect industrial predictive-maintenance data."
    }
    (out/"ch01_predictive_maintenance_provenance.json").write_text(json.dumps(meta,indent=2))

if __name__=="__main__":
    main()
