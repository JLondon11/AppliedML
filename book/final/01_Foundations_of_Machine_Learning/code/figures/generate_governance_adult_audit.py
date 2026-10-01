"""Generate a descriptive governance/fairness audit figure from the UCI Adult dataset.

This figure is intentionally descriptive. It reports group-conditional error and
calibration diagnostics on a historical census-derived benchmark; it does not
declare a model or group "fair" or "unfair" and does not prescribe a normative
fairness threshold.
"""
from __future__ import annotations
from pathlib import Path
import argparse, json, urllib.request, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, brier_score_loss, balanced_accuracy_score
from sklearn.calibration import calibration_curve

URL="https://archive.ics.uci.edu/static/public/2/adult.zip"
COLORS={"Female":"#7A6AA6","Male":"#355C7D"}

COLS=["age","workclass","fnlwgt","education","education_num","marital_status",
      "occupation","relationship","race","sex","capital_gain","capital_loss",
      "hours_per_week","native_country","income"]

def load(cache:Path):
    cache.parent.mkdir(parents=True,exist_ok=True)
    if not cache.exists():
        urllib.request.urlretrieve(URL,cache)
    with zipfile.ZipFile(cache) as z:
        def read(name):
            with z.open(name) as f:
                return pd.read_csv(f,names=COLS,skipinitialspace=True,na_values="?",
                                   comment="|",header=None)
        train=read("adult.data")
        test=read("adult.test")
    for d in (train,test):
        d["income"]=d["income"].astype(str).str.replace(".","",regex=False).str.strip()
        d.dropna(inplace=True)
    return train,test

def rates(y,p):
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    return {"fpr":fp/max(fp+tn,1),"fnr":fn/max(fn+tp,1),
            "tpr":tp/max(tp+fn,1),"tnr":tn/max(tn+fp,1)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="book/final/01_Foundations_of_Machine_Learning/figures")
    ap.add_argument("--cache",default=".cache/uci_adult.zip")
    args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    tr,te=load(Path(args.cache))
    ytr=(tr["income"]==">50K").astype(int).to_numpy()
    yte=(te["income"]==">50K").astype(int).to_numpy()

    # Sensitive audit columns are deliberately excluded from training features.
    drop=["income","sex","race"]
    Xtr=tr.drop(columns=drop); Xte=te.drop(columns=drop)
    num=["age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week"]
    cat=[c for c in Xtr.columns if c not in num]
    prep=ColumnTransformer([
        ("num",StandardScaler(),num),
        ("cat",OneHotEncoder(handle_unknown="ignore"),cat)
    ])
    model=make_pipeline(prep,LogisticRegression(max_iter=700,solver="liblinear",random_state=42))
    model.fit(Xtr,ytr); prob=model.predict_proba(Xte)[:,1]; pred=(prob>=.5).astype(int)

    rows=[]
    fig,axs=plt.subplots(1,2,figsize=(9.3,3.45))
    groups=["Female","Male"]
    fprs=[]; fnrs=[]
    for g in groups:
        mask=te["sex"].astype(str).str.strip().eq(g).to_numpy()
        r=rates(yte[mask],pred[mask])
        fprs.append(r["fpr"]); fnrs.append(r["fnr"])
        frac,mean=calibration_curve(yte[mask],prob[mask],n_bins=8,strategy="quantile")
        axs[1].plot(mean,frac,marker="o",lw=1.8,label=g,color=COLORS[g])
        rows.append({"group":g,"n":int(mask.sum()),"positive_rate":float(yte[mask].mean()),
                     "balanced_accuracy":balanced_accuracy_score(yte[mask],pred[mask]),
                     "false_positive_rate":r["fpr"],"false_negative_rate":r["fnr"],
                     "brier_score":brier_score_loss(yte[mask],prob[mask])})
    x=np.arange(2); w=.34
    axs[0].bar(x-w/2,fprs,width=w,label="false-positive rate",color="#4F7C6E")
    axs[0].bar(x+w/2,fnrs,width=w,label="false-negative rate",color="#B26E3B")
    axs[0].set_xticks(x,groups); axs[0].set_ylabel("Conditional error rate"); axs[0].set_ylim(0,1)
    axs[0].legend(frameon=False,fontsize=8)
    axs[1].plot([0,1],[0,1],"--",lw=1,color="#777777")
    axs[1].set_xlabel("Mean predicted probability"); axs[1].set_ylabel("Observed positive fraction")
    axs[1].set_xlim(0,1); axs[1].set_ylim(0,1); axs[1].legend(frameon=False,fontsize=8)
    for i,ax in enumerate(axs):
        ax.spines[["top","right"]].set_visible(False); ax.tick_params(direction="out")
        ax.text(.5,-.20,f"({chr(97+i)})",transform=ax.transAxes,ha="center",va="top",fontsize=10)
    fig.tight_layout(w_pad=2.1)
    for ext in ("png","svg","pdf"):
        fig.savefig(out/f"figure_01_15_governance_adult_audit.{ext}",
                    dpi=300 if ext=="png" else None,bbox_inches="tight",pad_inches=.03)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out/"ch01_governance_adult_audit.csv",index=False)
    prov={"source":"UCI Adult dataset","doi":"10.24432/C5XW20",
          "train_file":"adult.data","test_file":"adult.test",
          "audit_attribute":"sex","training_excluded_attributes":["sex","race"],
          "interpretation":"descriptive group-conditional diagnostics; no normative fairness threshold asserted"}
    (out/"ch01_governance_adult_provenance.json").write_text(json.dumps(prov,indent=2))

if __name__=="__main__":
    main()
