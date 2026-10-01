from pathlib import Path
import argparse, json, urllib.request, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

URL="https://archive.ics.uci.edu/static/public/400/crowdsourced+mapping.zip"

def find_label_col(df):
    for c in df.columns:
        if str(c).lower()=="class": return c
    return df.columns[-1]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="book/final/01_Foundations_of_Machine_Learning/figures")
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    zpath=out/"crowdsourced_mapping.zip"
    if not zpath.exists(): urllib.request.urlretrieve(URL,zpath)
    with zipfile.ZipFile(zpath) as z:
        names=z.namelist()
        tr=[n for n in names if n.lower().endswith("training.csv")][0]
        te=[n for n in names if n.lower().endswith("testing.csv")][0]
        with z.open(tr) as f: train=pd.read_csv(f)
        with z.open(te) as f: test=pd.read_csv(f)
    train.columns=[str(col).strip() for col in train.columns]
    test.columns=[str(col).strip() for col in test.columns]
    zpath.unlink(missing_ok=True)

    target=find_label_col(train)
    ytr=train[target].astype(str); yte=test[target].astype(str)
    Xtr=train.drop(columns=[target]); Xte=test.drop(columns=[target])
    # Restrict to numeric features; UCI file is numeric apart from class.
    Xtr=Xtr.apply(pd.to_numeric,errors="coerce").fillna(0)
    Xte=Xte.apply(pd.to_numeric,errors="coerce").fillna(0)
    common=[c for c in Xtr.columns if c in Xte.columns]
    print("COMMON_COLUMNS:", common)
    Xtr=Xtr[common]; Xte=Xte[common]

    ndvi=[c for c in common if str(c).lower()=="max_ndvi" or (str(c).endswith("_N") and str(c)[:8].isdigit())]
    maxndvi=[c for c in common if str(c).lower()=="max_ndvi"]
    if not ndvi:
        raise RuntimeError("No NDVI features found in UCI Crowdsourced Mapping schema")
    maxfeat=maxndvi if maxndvi else [ndvi[0]]

    models={
      "Max-NDVI logistic": (make_pipeline(StandardScaler(),LogisticRegression(max_iter=2500)),maxfeat),
      "NDVI-time-series logistic": (make_pipeline(StandardScaler(),LogisticRegression(max_iter=2500)),ndvi),
      "NDVI random forest": (RandomForestClassifier(n_estimators=400,max_depth=16,min_samples_leaf=2,class_weight="balanced",random_state=1729,n_jobs=-1),ndvi)
    }
    rows=[]; preds={}
    for name,(m,cols) in models.items():
        m.fit(Xtr[cols],ytr)
        p=m.predict(Xte[cols]); preds[name]=p
        rows.append({"model":name,"features":len(cols),
                     "accuracy":accuracy_score(yte,p),
                     "macro_f1":f1_score(yte,p,average="macro")})
    res=pd.DataFrame(rows); res.to_csv(out/"ch01_precision_agriculture_benchmark.csv",index=False)
    best=res.sort_values(["macro_f1","accuracy"],ascending=False).iloc[0]["model"]
    pred=preds[best]
    classes=sorted(yte.unique())

    # Mean NDVI temporal profiles by class using testing data.
    profile_cols=[c for c in ndvi if str(c).lower()!="max_ndvi"]
    if len(profile_cols)>24:
        # choose evenly spaced acquisition dates for legibility
        ix=np.linspace(0,len(profile_cols)-1,24).round().astype(int)
        profile_cols=[profile_cols[i] for i in ix]

    fig,axs=plt.subplots(1,2,figsize=(10.2,4.05),gridspec_kw={"width_ratios":[1.25,1]})
    palette=["#355C7D","#4F7C6E","#B26E3B","#7A6AA6","#657A8A","#8A6F5A"]
    for i,cl in enumerate(classes):
        m=(yte==cl).to_numpy()
        if m.sum()==0: continue
        prof=Xte.loc[m,profile_cols].mean(axis=0).to_numpy()
        axs[0].plot(range(len(profile_cols)),prof,lw=1.5,label=str(cl),color=palette[i%len(palette)])
    axs[0].set_xlabel("NDVI acquisition index")
    axs[0].set_ylabel("Mean NDVI")
    axs[0].legend(frameon=False,fontsize=7,ncol=2)

    cm=confusion_matrix(yte,pred,labels=classes,normalize="true")
    im=axs[1].imshow(cm,vmin=0,vmax=1,cmap="viridis")
    axs[1].set_xticks(range(len(classes)),classes,rotation=45,ha="right")
    axs[1].set_yticks(range(len(classes)),classes)
    axs[1].set_xlabel("Predicted class")
    axs[1].set_ylabel("True class")
    cb=fig.colorbar(im,ax=axs[1],fraction=.046,pad=.04)
    cb.set_label("Row-normalized fraction")
    for i,ax in enumerate(axs):
        if i==0: ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(direction="out")
        ax.text(.5,-.24,f"({chr(97+i)})",transform=ax.transAxes,ha="center",va="top",fontsize=10)
    fig.tight_layout(w_pad=1.8,pad=.25)
    for ext in ("png","svg","pdf"):
        fig.savefig(out/f"ch01_precision_agriculture_ndvi.{ext}",dpi=300 if ext=="png" else None,bbox_inches="tight",pad_inches=.02)
    plt.close(fig)

    meta={"dataset":"UCI Crowdsourced Mapping, id=400",
          "columns":common,
          "source_url":URL,
          "train_file_note":"UCI notes that training labels contain noise.",
          "test_file_note":"UCI states testing labels do not contain class-label errors.",
          "classes":classes,
          "best_model":best,
          "ndvi_feature_count":len(ndvi)}
    (out/"ch01_precision_agriculture_provenance.json").write_text(json.dumps(meta,indent=2))

if __name__=="__main__":
    main()
