"""Generate a real NASA C-MAPSS FD001 prognostics figure for Chapter 1.

Source: NASA Prognostics Center of Excellence, Turbofan Engine Degradation
Simulation Data Set. Uses train_FD001.txt, test_FD001.txt, RUL_FD001.txt.

The experiment fits a reproducible HistGradientBoostingRegressor on per-cycle
FD001 training rows with capped RUL targets and evaluates at the final observed
cycle of each FD001 test engine using NASA-provided true remaining life.
"""
from __future__ import annotations
from pathlib import Path
import argparse, io, json, urllib.request, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

URL="https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
COLORS={"prediction":"#B26E3B","identity":"#657A8A"}
TRAJECTORY_COLORS=["#355C7D","#4F7C6E","#B26E3B","#7A6AA6","#657A8A","#8B5E6B"]
SETTINGS=["setting1","setting2","setting3"]
SENSORS=[f"s{i}" for i in range(1,22)]
COLS=["unit","cycle",*SETTINGS,*SENSORS]
FEATURES=["cycle",*SETTINGS,"s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"]
RUL_CAP=125.0

def download(cache:Path):
    cache.parent.mkdir(parents=True,exist_ok=True)
    if not cache.exists():
        urllib.request.urlretrieve(URL,cache)
    return cache

def find_member(z,basename):
    matches=[n for n in z.namelist() if n.endswith(basename)]
    if not matches:
        raise RuntimeError(f"{basename} not found in archive")
    return matches[0]

def read_space_table(z,name,cols):
    raw=z.read(name)
    return pd.read_csv(io.BytesIO(raw),sep=r"\s+",header=None,names=cols,engine="python")

def load_fd001(cache:Path):
    # NASA's repository package may wrap the classic C-MAPSS files inside a
    # nested ZIP (for example CMAPSSData.zip). Support both layouts.
    outer=zipfile.ZipFile(download(cache))
    z=outer
    close_inner=False
    if not any(n.endswith("train_FD001.txt") for n in outer.namelist()):
        nested=[n for n in outer.namelist() if n.lower().endswith(".zip")]
        if not nested:
            raise RuntimeError("C-MAPSS text files and nested ZIP not found in NASA archive")
        raw=outer.read(nested[0])
        z=zipfile.ZipFile(io.BytesIO(raw))
        close_inner=True
    try:
        train=read_space_table(z,find_member(z,"train_FD001.txt"),COLS)
        test=read_space_table(z,find_member(z,"test_FD001.txt"),COLS)
        rul=read_space_table(z,find_member(z,"RUL_FD001.txt"),["rul"])
    finally:
        if close_inner:
            z.close()
        outer.close()
    return train,test,rul["rul"].to_numpy(dtype=float)

def add_train_rul(train):
    max_cycle=train.groupby("unit")["cycle"].transform("max")
    raw=(max_cycle-train["cycle"]).astype(float)
    out=train.copy()
    out["rul"]=np.minimum(raw,RUL_CAP)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="book/final/01_Foundations_of_Machine_Learning/figures")
    ap.add_argument("--cache",default=".cache/nasa_cmapss.zip")
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)

    train,test,true_rul=load_fd001(Path(args.cache))
    train_r=add_train_rul(train)

    model=HistGradientBoostingRegressor(
        learning_rate=0.055,max_iter=350,max_leaf_nodes=31,
        min_samples_leaf=30,l2_regularization=0.5,random_state=42
    )
    model.fit(train_r[FEATURES],train_r["rul"])

    last_idx=test.groupby("unit")["cycle"].idxmax()
    test_last=test.loc[last_idx].sort_values("unit")
    pred=np.clip(model.predict(test_last[FEATURES]),0,None)
    rmse=float(mean_squared_error(true_rul,pred)**0.5)
    mae=float(mean_absolute_error(true_rul,pred))

    pd.DataFrame({
        "unit":test_last["unit"].to_numpy(dtype=int),
        "last_observed_cycle":test_last["cycle"].to_numpy(dtype=int),
        "true_rul_cycles":true_rul,
        "predicted_rul_cycles":pred,
        "error_cycles":pred-true_rul
    }).to_csv(out/"ch01_cmapss_fd001_predictions.csv",index=False)

    # Panel (a): real sensor trajectories from six representative training engines.
    # Sensor 11 varies with degradation in FD001; normalize per trajectory only
    # for visual comparison while preserving the actual recorded cycle ordering.
    units=[1,20,40,60,80,100]
    fig,axs=plt.subplots(1,2,figsize=(10.4,3.55))
    for u,fc in zip(units,TRAJECTORY_COLORS):
        d=train[train["unit"]==u]
        x=d["cycle"].to_numpy()/d["cycle"].max()
        y=d["s11"].to_numpy(dtype=float)
        y=(y-y.mean())/max(y.std(),1e-12)
        axs[0].plot(x,y,lw=1.1,alpha=.82,color=fc,label=f"engine {u}")
    axs[0].set_xlabel("Fraction of observed run-to-failure life")
    axs[0].set_ylabel("Sensor 11 (per-engine standardized)")
    axs[0].legend(frameon=False,fontsize=7,ncol=2)

    hi=max(float(np.max(true_rul)),float(np.max(pred)))+5
    axs[1].scatter(true_rul,pred,s=24,alpha=.72,color=COLORS["prediction"],
                   edgecolor="white",linewidth=.35)
    axs[1].plot([0,hi],[0,hi],"--",lw=1.2,color=COLORS["identity"])
    axs[1].set_xlim(0,hi); axs[1].set_ylim(0,hi)
    axs[1].set_xlabel("NASA true remaining life (cycles)")
    axs[1].set_ylabel("Predicted remaining life (cycles)")
    axs[1].text(.04,.93,f"RMSE = {rmse:.2f} cycles\nMAE = {mae:.2f} cycles",
                transform=axs[1].transAxes,va="top",fontsize=8)

    for i,ax in enumerate(axs):
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(direction="out")
        ax.text(.5,-.20,f"({chr(97+i)})",transform=ax.transAxes,
                ha="center",va="top",fontsize=10)
    fig.tight_layout(w_pad=2.0)
    for ext in ("png","svg","pdf"):
        fig.savefig(out/f"ch01_turbofan_cmapss_fd001.{ext}",
                    dpi=300 if ext=="png" else None,bbox_inches="tight",pad_inches=.03)
    plt.close(fig)

    prov={
        "source":"NASA Prognostics Center of Excellence Turbofan Engine Degradation Simulation Data Set",
        "source_url":URL,
        "subset":"FD001",
        "train_engines":int(train["unit"].nunique()),
        "test_engines":int(test["unit"].nunique()),
        "conditions":"one operating condition",
        "fault_modes":"one fault mode (HPC degradation)",
        "target":"training RUL = min(max_cycle_per_engine - cycle, 125); test target from NASA RUL_FD001.txt at each engine final observed cycle",
        "features":FEATURES,
        "model":"sklearn HistGradientBoostingRegressor",
        "random_state":42,
        "rmse_cycles":rmse,
        "mae_cycles":mae,
        "figure_panel_a":"actual FD001 train_FD001 sensor 11 trajectories for engines 1,20,40,60,80,100; per-engine standardized only for plotting",
        "figure_panel_b":"NASA true RUL_FD001 values versus predictions at final observed test cycles"
    }
    (out/"ch01_turbofan_cmapss_fd001_provenance.json").write_text(json.dumps(prov,indent=2))
    print(json.dumps({"rmse":rmse,"mae":mae,"test_engines":len(true_rul)},indent=2))

if __name__=="__main__":
    main()
