from pathlib import Path
import argparse, json, urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, accuracy_score, f1_score,
    precision_recall_curve, precision_score, recall_score
)
from skimage.feature import hog

URL="https://zenodo.org/records/10519652/files/pneumoniamnist.npz?download=1"
COLORS={"Pixel Logistic":"#355C7D","HOG Logistic":"#4F7C6E","Random Forest":"#B26E3B"}

def best_threshold(y,s):
    p,r,t=precision_recall_curve(y,s)
    f=2*p*r/np.maximum(p+r,1e-12)
    i=int(np.nanargmax(f[:-1]))
    return float(t[i])

def hog_features(arr):
    return np.vstack([hog(im,pixels_per_cell=(4,4),cells_per_block=(2,2),orientations=9,feature_vector=True)
                      for im in arr])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="book/final/01_Foundations_of_Machine_Learning/figures")
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    cache=out/"pneumoniamnist.npz"
    if not cache.exists():
        urllib.request.urlretrieve(URL,cache)
    z=np.load(cache)
    Xtr=z["train_images"].astype(np.float32)/255.0
    ytr=z["train_labels"].reshape(-1).astype(int)
    Xv=z["val_images"].astype(np.float32)/255.0
    yv=z["val_labels"].reshape(-1).astype(int)
    Xte=z["test_images"].astype(np.float32)/255.0
    yte=z["test_labels"].reshape(-1).astype(int)

    flat_tr=Xtr.reshape(len(Xtr),-1); flat_v=Xv.reshape(len(Xv),-1); flat_te=Xte.reshape(len(Xte),-1)
    htr=hog_features(Xtr); hv=hog_features(Xv); hte=hog_features(Xte)

    models={
      "Pixel Logistic": (make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,C=.5)),flat_tr,flat_v,flat_te),
      "HOG Logistic": (make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,C=1.0)),htr,hv,hte),
      "Random Forest": (RandomForestClassifier(n_estimators=450,max_depth=14,min_samples_leaf=2,class_weight="balanced",random_state=1729,n_jobs=-1),flat_tr,flat_v,flat_te)
    }
    rows=[]; curves={}
    for name,(m,a,b,c) in models.items():
        m.fit(a,ytr)
        sv=m.predict_proba(b)[:,1]; st=m.predict_proba(c)[:,1]
        th=best_threshold(yv,sv)
        pred=(st>=th).astype(int)
        rows.append({
          "model":name,
          "roc_auc":roc_auc_score(yte,st),
          "average_precision":average_precision_score(yte,st),
          "accuracy":accuracy_score(yte,pred),
          "f1":f1_score(yte,pred),
          "precision":precision_score(yte,pred,zero_division=0),
          "recall":recall_score(yte,pred,zero_division=0),
          "val_selected_threshold":th
        })
        p,r,_=precision_recall_curve(yte,st); curves[name]=(p,r)
    pd.DataFrame(rows).to_csv(out/"ch01_medical_pneumonia_benchmark.csv",index=False)

    mean0=Xte[yte==0].mean(axis=0)
    mean1=Xte[yte==1].mean(axis=0)

    fig,axs=plt.subplots(1,3,figsize=(10.2,3.35),gridspec_kw={"width_ratios":[1,1,1.45]})
    axs[0].imshow(mean0,cmap="gray",vmin=0,vmax=1)
    axs[1].imshow(mean1,cmap="gray",vmin=0,vmax=1)
    for ax in axs[:2]:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
    for name,(p,r) in curves.items():
        axs[2].plot(r,p,lw=2,label=name,color=COLORS[name])
    prev=float(yte.mean())
    axs[2].axhline(prev,ls="--",lw=1,color="#777777",label=f"Test prevalence ({prev:.1%})")
    axs[2].set_xlabel("Recall")
    axs[2].set_ylabel("Precision")
    axs[2].set_xlim(0,1); axs[2].set_ylim(0,1)
    axs[2].legend(frameon=False,fontsize=7)
    axs[2].spines[["top","right"]].set_visible(False)
    axs[2].tick_params(direction="out")
    for i,ax in enumerate(axs):
        ax.text(.5,-.16,f"({chr(97+i)})",transform=ax.transAxes,ha="center",va="top",fontsize=10)
    fig.tight_layout(w_pad=1.2,pad=.25)
    for ext in ("png","svg","pdf"):
        fig.savefig(out/f"ch01_medical_pneumoniamnist.{ext}",dpi=300 if ext=="png" else None,bbox_inches="tight",pad_inches=.02)
    plt.close(fig)

    meta={
      "dataset":"MedMNIST v2 PneumoniaMNIST",
      "url":URL,
      "train":int(len(Xtr)),"validation":int(len(Xv)),"test":int(len(Xte)),
      "image_shape":list(Xtr.shape[1:]),
      "task":"binary chest-X-ray classification: normal versus pneumonia",
      "clinical_use":"Not intended for clinical use; educational/research benchmark."
    }
    (out/"ch01_medical_pneumonia_provenance.json").write_text(json.dumps(meta,indent=2))
    # Avoid retaining the downloaded source archive in the final figure directory.
    cache.unlink(missing_ok=True)

if __name__=="__main__":
    main()
