"""Generate the Chapter 1 predictive-maintenance evidence from real CWRU bearing data.

Sources:
- CWRU normal baseline file 97.mat (0 hp)
- CWRU 12 kHz drive-end inner-race fault file 105.mat (0.007 in, 0 hp)

The experiment compares blocked time windows from the two recordings. It is a
controlled diagnostic demonstration, not an estimate of field failure rates.
"""
from __future__ import annotations
from pathlib import Path
import argparse, json, urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import kurtosis
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

BASE="https://engineering.case.edu/sites/default/files"
FILES={
    "normal_0":"97.mat","normal_1":"98.mat","normal_2":"99.mat","normal_3":"100.mat",
    "inner_race_007_0":"105.mat","inner_race_007_1":"106.mat",
    "inner_race_007_2":"107.mat","inner_race_007_3":"108.mat"
}
FS=12000
FS_NORMAL=48000
FS_FAULT=12000
COLORS={"normal":"#355C7D","fault":"#B26E3B"}

def download(cache:Path):
    cache.mkdir(parents=True,exist_ok=True)
    paths={}
    for label,name in FILES.items():
        p=cache/name
        if not p.exists():
            urllib.request.urlretrieve(f"{BASE}/{name}",p)
        paths[label]=p
    return paths

def de_signal(path:Path):
    d=loadmat(path)
    stem=path.stem
    # CWRU files can contain variables copied from neighboring recordings
    # (notably 99.mat contains X098_* and X099_*). Prefer the variable whose
    # numeric prefix matches the file identifier.
    expected=f"X{int(stem):03d}_DE_time" if stem.isdigit() else None
    if expected and expected in d:
        key=expected
    else:
        keys=[k for k in d if k.upper().endswith("DE_TIME")]
        if not keys:
            keys=[k for k,v in d.items() if not k.startswith("__") and hasattr(v,"shape") and np.asarray(v).size>10000]
        if not keys:
            raise RuntimeError(f"No vibration vector found in {path}")
        key=keys[0]
    x=np.asarray(d[key]).reshape(-1).astype(float)
    return x,key

def windows(x,n=2048,hop=2048):
    return np.stack([x[i:i+n] for i in range(0,len(x)-n+1,hop)])

def feats(W):
    out=[]
    for w in W:
        w=w-np.mean(w)
        rms=float(np.sqrt(np.mean(w*w)))
        peak=float(np.max(np.abs(w)))
        crest=peak/max(rms,1e-12)
        kur=float(kurtosis(w,fisher=False,bias=False))
        spec=np.abs(np.fft.rfft(w))
        freqs=np.fft.rfftfreq(len(w),1/FS)
        centroid=float(np.sum(freqs*spec)/max(np.sum(spec),1e-12))
        band_hi=float(np.sum(spec[freqs>=2000]**2)/max(np.sum(spec**2),1e-12))
        out.append([rms,crest,kur,centroid,band_hi])
    return np.asarray(out)

def stack_files(feature_map, labels):
    X=[]; y=[]
    for name,target in labels:
        X.append(feature_map[name])
        y.append(np.full(len(feature_map[name]),target,dtype=int))
    return np.vstack(X),np.concatenate(y)

def metrics(y,p):
    pred=(p>=.5).astype(int)
    return balanced_accuracy_score(y,pred),roc_auc_score(y,p)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="book/final/01_Foundations_of_Machine_Learning/figures")
    ap.add_argument("--cache",default=".cache/cwru")
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    paths=download(Path(args.cache))
    signals={}; keys={}; feature_map={}
    for name,path in paths.items():
        sig,key=de_signal(path)
        # The normal baseline recordings are 48 kHz while this 12-kHz drive-end
        # fault subset is 12 kHz. Downsample normal data by four before any
        # time/frequency or feature comparison.
        if name.startswith("normal_"):
            sig=resample_poly(sig,up=1,down=4)
        signals[name]=sig; keys[name]=key
        feature_map[name]=feats(windows(sig))

    # File-level domain holdout: train on 0, 1, and 2 hp recordings; evaluate
    # exclusively on the unseen 3 hp recordings. This prevents windows from the
    # same recording from appearing in both train and test sets.
    train_labels=[
        ("normal_0",0),("normal_1",0),("normal_2",0),
        ("inner_race_007_0",1),("inner_race_007_1",1),("inner_race_007_2",1)
    ]
    test_labels=[("normal_3",0),("inner_race_007_3",1)]
    Xtr,ytr=stack_files(feature_map,train_labels)
    Xte,yte=stack_files(feature_map,test_labels)

    models={
      "Logistic regression":make_pipeline(StandardScaler(),LogisticRegression(max_iter=1500,random_state=42)),
      "Random forest":RandomForestClassifier(n_estimators=300,max_depth=7,min_samples_leaf=3,random_state=42)
    }
    rows=[]
    for name,m in models.items():
        m.fit(Xtr,ytr); p=m.predict_proba(Xte)[:,1]
        ba,auc=metrics(yte,p)
        rows.append({"model":name,"balanced_accuracy":ba,"roc_auc":auc,
                     "train_windows":len(ytr),"test_windows":len(yte),
                     "train_recordings":6,"test_recordings":2,
                     "test_load_hp":3})
    pd.DataFrame(rows).to_csv(out/"ch01_predictive_maintenance_benchmark.csv",index=False)

    fig,axs=plt.subplots(1,3,figsize=(10.6,3.15))
    normal=signals["normal_3"]; fault=signals["inner_race_007_3"]
    nshow=2400; t=np.arange(nshow)/FS
    axs[0].plot(t,normal[:nshow],lw=.8,color=COLORS["normal"],label="normal, held-out 3 hp")
    axs[0].plot(t,fault[:nshow],lw=.8,color=COLORS["fault"],alpha=.8,label="inner-race fault, held-out 3 hp")
    axs[0].set_xlabel("Time (s)"); axs[0].set_ylabel("Drive-end acceleration (recorded units)")
    axs[0].legend(frameon=False,fontsize=7)

    for x,label,c0 in [(normal,"normal",COLORS["normal"]),(fault,"fault",COLORS["fault"])]:
        seg=x[:8192]-np.mean(x[:8192]); sp=np.abs(np.fft.rfft(seg)); fr=np.fft.rfftfreq(len(seg),1/FS)
        axs[1].plot(fr,sp/max(sp.max(),1e-12),lw=1.0,label=label,color=c0)
    axs[1].set_xlim(0,4000); axs[1].set_xlabel("Frequency (Hz)"); axs[1].set_ylabel("Normalized magnitude")
    axs[1].legend(frameon=False,fontsize=7)

    F0=feature_map["normal_3"]; F1=feature_map["inner_race_007_3"]
    axs[2].scatter(F0[:,0],F0[:,2],s=12,alpha=.6,color=COLORS["normal"],label="normal")
    axs[2].scatter(F1[:,0],F1[:,2],s=12,alpha=.6,color=COLORS["fault"],label="fault")
    axs[2].set_xlabel("Window RMS"); axs[2].set_ylabel("Window kurtosis")
    axs[2].legend(frameon=False,fontsize=7)
    for i,ax in enumerate(axs):
        ax.spines[["top","right"]].set_visible(False); ax.tick_params(direction="out")
        ax.text(.5,-.23,f"({chr(97+i)})",transform=ax.transAxes,ha="center",va="top",fontsize=10)
    fig.tight_layout(w_pad=1.8)
    for ext in ("png","svg","pdf"):
        fig.savefig(out/f"figure_01_24_predictive_maintenance_cwru.{ext}",dpi=300 if ext=="png" else None,bbox_inches="tight",pad_inches=.03)
    plt.close(fig)

    provenance={
      "source":"Case Western Reserve University Bearing Data Center",
      "normal_files":["97.mat","98.mat","99.mat","100.mat"],
      "fault_files":["105.mat","106.mat","107.mat","108.mat"],
      "variables":keys,
      "analysis_sampling_rate_hz":FS,
      "normal_original_sampling_rate_hz":FS_NORMAL,
      "fault_original_sampling_rate_hz":FS_FAULT,
      "normal_resampling":"polyphase downsample by 4 to 12 kHz before feature extraction",
      "window_samples":2048,"hop_samples":2048,
      "split":"file-level operating-condition holdout: train on 0/1/2 hp normal and 0.007-in inner-race-fault recordings; test only on unseen 3 hp normal/fault recordings",
      "limitations":"Controlled normal-versus-inner-race-fault diagnostic using one fault size; not a field prevalence or fleet reliability estimate."
    }
    (out/"ch01_predictive_maintenance_provenance.json").write_text(json.dumps(provenance,indent=2))

if __name__=="__main__":
    main()
