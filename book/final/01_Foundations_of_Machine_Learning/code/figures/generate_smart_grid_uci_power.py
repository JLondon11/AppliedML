"""Generate Chapter 1 smart-grid/load-forecasting evidence from the UCI household power dataset."""
from __future__ import annotations
from pathlib import Path
import argparse, io, json, urllib.request, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt\nimport matplotlib.dates as mdates
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

URL="https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
COLORS={"actual":"#355C7D","persistence":"#657A8A","seasonal":"#B26E3B","ridge":"#4F7C6E"}

def load_hourly(cache:Path):
    cache.parent.mkdir(parents=True,exist_ok=True)
    if not cache.exists():
        urllib.request.urlretrieve(URL,cache)
    with zipfile.ZipFile(cache) as z:
        name=[n for n in z.namelist() if n.endswith("household_power_consumption.txt")][0]
        with z.open(name) as f:
            df=pd.read_csv(f,sep=";",na_values="?",low_memory=False)
    ts=pd.to_datetime(df["Date"]+" "+df["Time"],format="%d/%m/%Y %H:%M:%S")
    s=pd.Series(pd.to_numeric(df["Global_active_power"],errors="coerce").to_numpy(),index=ts,name="kw")
    hourly=s.resample("1h").mean().interpolate(limit=3)
    return hourly.dropna()

def supervised(s,lags=(1,2,24,25,168)):
    df=pd.DataFrame({"y":s})
    for l in lags: df[f"lag_{l}"]=s.shift(l)
    return df.dropna()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="book/final/01_Foundations_of_Machine_Learning/figures")
    ap.add_argument("--cache",default=".cache/uci_household_power.zip")
    args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    s=load_hourly(Path(args.cache))
    df=supervised(s)
    split=int(len(df)*.80); tr=df.iloc[:split]; te=df.iloc[split:]
    features=[c for c in df.columns if c.startswith("lag_")]
    ridge=Ridge(alpha=1.0).fit(tr[features],tr["y"])
    pred_ridge=ridge.predict(te[features])
    pred_persist=te["lag_1"].to_numpy()
    pred_seasonal=te["lag_24"].to_numpy()
    rows=[]
    for name,p in [("Persistence (1 h lag)",pred_persist),("Seasonal naive (24 h lag)",pred_seasonal),("Ridge lag model",pred_ridge)]:
        rows.append({"model":name,"mae_kw":mean_absolute_error(te["y"],p),
                     "rmse_kw":mean_squared_error(te["y"],p)**0.5,
                     "train_hours":len(tr),"test_hours":len(te)})
    pd.DataFrame(rows).to_csv(out/"ch01_smart_grid_benchmark.csv",index=False)

    n=24*7
    y=te["y"].iloc[:n]
    fig,axs=plt.subplots(1,2,figsize=(10.2,3.35))
    axs[0].plot(y.index,y.to_numpy(),lw=1.5,color=COLORS["actual"],label="actual")
    axs[0].plot(y.index,pred_seasonal[:n],lw=1.1,color=COLORS["seasonal"],label="24 h seasonal naive")
    axs[0].plot(y.index,pred_ridge[:n],lw=1.1,color=COLORS["ridge"],label="ridge lag model")
    axs[0].set_ylabel("Hourly mean active power (kW)"); axs[0].set_xlabel("Time")
    axs[0].tick_params(axis="x",rotation=25); axs[0].legend(frameon=False,fontsize=7)

    abs_err=[np.abs(te["y"].to_numpy()-p) for p in (pred_persist,pred_seasonal,pred_ridge)]
    bp=axs[1].boxplot(abs_err,tick_labels=["1 h\npersistence","24 h\nseasonal","ridge\nlags"],
                       showfliers=False,patch_artist=True)
    for patch,fc in zip(bp["boxes"],["#DDE5EB","#EFE0D4","#DCE8E2"]):
        patch.set_facecolor(fc)
        patch.set_edgecolor("#555555")
    for med in bp["medians"]:
        med.set_color("#B26E3B")
        med.set_linewidth(1.7)
    axs[1].set_ylabel("Absolute forecast error (kW)")
    for i,ax in enumerate(axs):
        ax.spines[["top","right"]].set_visible(False); ax.tick_params(direction="out")
        ax.text(.5,-.20,f"({chr(97+i)})",transform=ax.transAxes,ha="center",va="top",fontsize=10)
    fig.tight_layout(w_pad=2.0)
    for ext in ("png","svg","pdf"):
        fig.savefig(out/f"figure_01_30_smart_grid_uci_power.{ext}",dpi=300 if ext=="png" else None,bbox_inches="tight",pad_inches=.03)
    plt.close(fig)

    prov={
      "source":"UCI Individual Household Electric Power Consumption, dataset 235",
      "doi":"10.24432/C58K54",
      "raw_sampling":"1 minute","aggregation":"hourly mean Global_active_power",
      "missing_handling":"numeric '?' -> missing; hourly mean; interpolation limited to gaps <=3 hours; remaining missing rows dropped",
      "split":"first 80% chronological supervised rows train, final 20% test",
      "lags_hours":[1,2,24,25,168]
    }
    (out/"ch01_smart_grid_provenance.json").write_text(json.dumps(prov,indent=2))

if __name__=="__main__":
    main()
