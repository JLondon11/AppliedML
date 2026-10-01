from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

COLORS={"Actual":"#243447","Persistence (24 h)":"#7A6AA6","Ridge":"#4F7C6E","HistGBR":"#B26E3B"}

def metrics(y, p):
    mae=mean_absolute_error(y,p)
    rmse=mean_squared_error(y,p)**0.5
    denom=np.maximum(np.abs(y),0.1)
    mape=np.mean(np.abs(y-p)/denom)*100
    return mae,rmse,mape

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="book/final/01_Foundations_of_Machine_Learning/figures")
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)

    d=fetch_ucirepo(id=235)
    X=d.data.features.copy()

    # UCI loader exposes Date/Time and numeric fields. Construct timestamp robustly.
    if "Date" in X.columns and "Time" in X.columns:
        ts=pd.to_datetime(X["Date"].astype(str)+" "+X["Time"].astype(str),dayfirst=True,errors="coerce")
    else:
        raise RuntimeError("Expected Date and Time columns in UCI household power dataset")

    s=pd.to_numeric(X["Global_active_power"],errors="coerce")
    df=pd.DataFrame({"timestamp":ts,"load":s}).dropna().set_index("timestamp").sort_index()
    hourly=df["load"].resample("1h").mean().interpolate(limit=3).dropna()

    # Use the final 18 months to keep the benchmark efficient while preserving seasonality.
    end=hourly.index.max()
    start=end-pd.Timedelta(days=548)
    h=hourly.loc[hourly.index>=start].copy()
    frame=pd.DataFrame({"y":h})
    frame["lag1"]=h.shift(1)
    frame["lag24"]=h.shift(24)
    frame["lag168"]=h.shift(168)
    frame["roll24"]=h.shift(1).rolling(24).mean()
    frame["hour_sin"]=np.sin(2*np.pi*frame.index.hour/24)
    frame["hour_cos"]=np.cos(2*np.pi*frame.index.hour/24)
    frame["dow_sin"]=np.sin(2*np.pi*frame.index.dayofweek/7)
    frame["dow_cos"]=np.cos(2*np.pi*frame.index.dayofweek/7)
    frame=frame.dropna()

    split=int(len(frame)*0.80)
    train=frame.iloc[:split]; test=frame.iloc[split:]
    feats=[c for c in frame.columns if c!="y"]
    Xtr=train[feats].to_numpy(); ytr=train["y"].to_numpy()
    Xte=test[feats].to_numpy(); yte=test["y"].to_numpy()

    preds={"Persistence (24 h)":test["lag24"].to_numpy()}
    ridge=Ridge(alpha=3.0).fit(Xtr,ytr)
    preds["Ridge"]=ridge.predict(Xte)
    hgb=HistGradientBoostingRegressor(max_depth=6,learning_rate=.06,max_iter=260,l2_regularization=.1,random_state=1729).fit(Xtr,ytr)
    preds["HistGBR"]=hgb.predict(Xte)

    rows=[]
    for name,p in preds.items():
        mae,rmse,mape=metrics(yte,p)
        rows.append({"model":name,"mae_kw":mae,"rmse_kw":rmse,"mape_percent":mape})
    pd.DataFrame(rows).to_csv(out/"ch01_smart_grid_forecast_benchmark.csv",index=False)

    # Final 7 days for visual forecast.
    n=min(24*7,len(test))
    t=test.index[-n:]; ya=yte[-n:]
    fig,axs=plt.subplots(1,2,figsize=(10.2,4.05))
    axs[0].plot(t,ya,lw=2,color=COLORS["Actual"],label="Actual")
    for name in ["Persistence (24 h)","Ridge","HistGBR"]:
        axs[0].plot(t,preds[name][-n:],lw=1.25,color=COLORS[name],label=name)
    axs[0].set_ylabel("Hourly active power (kW)")
    axs[0].set_xlabel("Time")
    axs[0].tick_params(axis="x",rotation=25)
    axs[0].legend(frameon=False,fontsize=7,ncol=2)

    # RMSE by hour of day for strongest model and persistence.
    hrs=test.index.hour.to_numpy()
    for name in ["Persistence (24 h)","HistGBR"]:
        vals=[]
        for hr in range(24):
            m=hrs==hr
            vals.append(np.sqrt(np.mean((yte[m]-preds[name][m])**2)) if m.any() else np.nan)
        axs[1].plot(range(24),vals,marker="o",ms=3,lw=1.6,label=name,color=COLORS[name])
    axs[1].set_xlabel("Hour of day")
    axs[1].set_ylabel("RMSE (kW)")
    axs[1].set_xticks([0,4,8,12,16,20,23])
    axs[1].legend(frameon=False,fontsize=8)

    for i,ax in enumerate(axs):
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(direction="out")
        ax.text(.5,-.22,f"({chr(97+i)})",transform=ax.transAxes,ha="center",va="top",fontsize=10)
    fig.tight_layout(w_pad=2.0,pad=.3)
    for ext in ("png","svg","pdf"):
        fig.savefig(out/f"ch01_smart_grid_load_forecasting.{ext}",dpi=300 if ext=="png" else None,bbox_inches="tight",pad_inches=.02)
    plt.close(fig)

    meta={
      "dataset":"UCI Individual Household Electric Power Consumption, id=235",
      "raw_instances":2075259,
      "period":"December 2006 to November 2010; final 18 months used after hourly aggregation",
      "split":"chronological 80/20 after lag construction",
      "features":"lags 1, 24, 168 h; lagged 24 h rolling mean; hour/day-of-week cyclical features"
    }
    (out/"ch01_smart_grid_forecast_provenance.json").write_text(json.dumps(meta,indent=2))

if __name__=="__main__":
    main()
