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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

BASE="https://engineering.case.edu/sites/default/files"
FILES={"normal_0":"97.mat","inner_race_007_0":"105.mat"}
FS=12000
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
    keys=[k for k in d if k.upper().endswith("DE_TIME")]
    if not keys:
        keys=[k for k,v in d.items() if not k.startswith("__") and hasattr(v,"shape") and np.asarray(v).size>10000]
    if not keys:
        raise RuntimeError(f"No vibration vector found in {path}")
    x=np.asarray(d[keys[0]]).reshape(-1).astype(float)
    return x,keys[0]

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

def blocked_split(F0,F1,frac=.70):
    n0=int(len(F0)*frac); n1=int(len(F1)*frac)
    Xtr=np.vstack([F0[:n0],F1[:n1]])
    ytr=np.r_[np.zeros(n0,int),np.ones(n1,int)]
    Xte=np.vstack([F0[n0:],F1[n1:]])
    yte=np.r_[np.zeros(len(F0)-n0,int),np.ones(len(F1)-n1,int)]
    return Xtr,ytr,Xte,yte

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
    normal,key0=de_signal(paths["normal_0"]); fault,key1=de_signal(paths["inner_race_007_0"])
    W0,W1=windows(normal),windows(fault)
    F0,F1=feats(W0),feats(W1)
    Xtr,ytr,Xte,yte=blocked_split(F0,F1)

    models={
      "Logistic regression":make_pipeline(StandardScaler(),LogisticRegression(max_iter=1500,random_state=42)),
      "Random forest":RandomForestClassifier(n_estimators=200,max_depth=6,min_samples_leaf=3,random_state=42)
    }
    rows=[]
    for name,m in models.items():
        m.fit(Xtr,ytr); p=m.predict_proba(Xte)[:,1]
        ba,auc=metrics(yte,p)
        rows.append({"model":name,"balanced_accuracy":ba,"roc_auc":auc,
                     "train_windows":len(ytr),"test_windows":len(yte)})
    pd.DataFrame(rows).to_csv(out/"ch01_predictive_maintenance_benchmark.csv",index=False)

    fig,axs=plt.subplots(1,3,figsize=(10.6,3.15))
    nshow=2400; t=np.arange(nshow)/FS
    axs[0].plot(t,normal[:nshow],lw=.8,color=COLORS["normal"],label="normal")
    axs[0].plot(t,fault[:nshow],lw=.8,color=COLORS["fault"],alpha=.8,label="inner-race fault")
    axs[0].set_xlabel("Time (s)"); axs[0].set_ylabel("Drive-end acceleration (recorded units)")
    axs[0].legend(frameon=False,fontsize=7)

    for x,label,c in [(normal,"normal",COLORS["normal"]),(fault,"fault",COLORS["fault"])]:
        seg=x[:8192]-np.mean(x[:8192]); sp=np.abs(np.fft.rfft(seg)); fr=np.fft.rfftfreq(len(seg),1/FS)
        axs[1].plot(fr,sp/max(sp.max(),1e-12),lw=1.0,label=label,color=c)
    axs[1].set_xlim(0,4000); axs[1].set_xlabel("Frequency (Hz)"); axs[1].set_ylabel("Normalized magnitude")
    axs[1].legend(frameon=False,fontsize=7)

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
      "normal_file":"97.mat","fault_file":"105.mat",
      "normal_variable":key0,"fault_variable":key1,
      "sampling_rate_hz":FS,"window_samples":2048,"hop_samples":2048,
      "split":"blocked first 70% windows train, final 30% windows test independently within each recording",
      "limitations":"Two 0-hp recordings; diagnostic demonstration, not a field prevalence or fleet reliability estimate."
    }
    (out/"ch01_predictive_maintenance_provenance.json").write_text(json.dumps(provenance,indent=2))

if __name__=="__main__":
    main()
