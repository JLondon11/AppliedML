"""Scientific AI Figure 2 from authentic NGDC annual sunspot observations.
No fabricated measurements, simulated fields, or artificially noisy observations.
Usage: python figure_02_scientific_discovery.py --output-dir figures
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.datasets import sunspots

def main(out):
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    data=sunspots.load_pandas().data
    years=data.YEAR.to_numpy(dtype=int)
    values=data.SUNACTIVITY.to_numpy(dtype=np.float64)
    assert len(values)==309 and years[0]==1700 and years[-1]==2008
    assert np.isfinite(values).all() and np.all(np.diff(years)==1)
    L=12
    X=np.stack([values[i-L:i] for i in range(L,len(values))])
    y=values[L:]; ty=years[L:]
    train=ty<=1889
    val=(ty>=1890)&(ty<=1949)
    test=ty>=1950
    assert tuple(map(int,(train.sum(),val.sum(),test.sum())))==(178,60,59)
    mu=X[train].mean(0); sd=X[train].std(0);sd[sd==0]=1
    Xt=(X[train]-mu)/sd
    Xv=(X[val]-mu)/sd
    Xh=(X[test]-mu)/sd
    tuning=[]
    for alpha in [0.1,1.,10.,100.,1000.]:
        r=Ridge(alpha=alpha).fit(Xt,y[train])
        e=float(np.sqrt(mean_squared_error(y[val],r.predict(Xv))))
        tuning.append({"alpha":alpha,"validation_rmse":e})
    chosen=min(tuning,key=lambda z:z["validation_rmse"])["alpha"]
    model=Ridge(alpha=chosen).fit(Xt,y[train])
    pred=model.predict(Xh); persistence=X[test,-1]
    rmse=float(np.sqrt(mean_squared_error(y[test],pred)))
    base=float(np.sqrt(mean_squared_error(y[test],persistence)))
    assert np.isfinite(rmse) and np.isfinite(base)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
        "axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(2,2,figsize=(12.2,7.7),constrained_layout=True)
    a,b,c,d=axes.flat
    a.plot(years,values,color="#315b79",lw=1.25)
    a.set(title="(a) Recorded annual sunspot numbers",xlabel="Year",ylabel="Sunspot number")
    avg=np.convolve(values,np.ones(11)/11,mode="valid")
    yavg=np.convolve(years,np.ones(11)/11,mode="valid")
    b.plot(years,values,color="#b9c5ce",lw=.8,label="Observed annual values")
    b.plot(yavg,avg,color="#af673c",lw=2.2,label="Observed 11-year rolling average")
    b.set(title="(b) Recorded temporal structure",xlabel="Year",ylabel="Sunspot number")
    b.legend(frameon=False,fontsize=8)
    c.scatter(values[:-1],values[1:],s=10,alpha=.6,color="#387c71")
    c.set(title="(c) Observed one-year lag relationship",
          xlabel="Observed sunspots at year t - 1",ylabel="Observed sunspots at year t")
    c.set_xlim(left=0);c.set_ylim(bottom=0)
    d.plot(ty[test],y[test],color="#243f59",lw=2,label="Observed sunspots, held-out")
    d.plot(ty[test],pred,color="#b66b43",lw=1.7,
           label=f"Fitted Ridge forecast (RMSE {rmse:.2f})")
    d.plot(ty[test],persistence,color="#678777",lw=1,alpha=.65,
           label=f"Persistence (RMSE {base:.2f})")
    d.set(title="(d) Real-data forecast evaluation (1950-2008)",
          xlabel="Held-out target year",ylabel="Sunspot number")
    d.legend(frameon=False,fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.15)
    for ext in ["svg","png","pdf"]:
        fig.savefig(out/f"figure_02_evolution_scientific_discovery.{ext}",
                    dpi=220,facecolor="white")
    plt.close(fig)
    result={"dataset":"NGDC annual sunspots from statsmodels.datasets.sunspots",
      "dataset_url":"https://www.statsmodels.org/stable/datasets/generated/sunspots.html",
      "dataset_observations":309,"dataset_years":[1700,2008],
      "dataset_values_float64_sha256":hashlib.sha256(values.tobytes()).hexdigest(),
      "lookback":12,"training_windows":int(train.sum()),
      "validation_windows":int(val.sum()),"testing_windows":int(test.sum()),
      "train_target_years":[1712,1889],"validation_target_years":[1890,1949],
      "test_target_years":[1950,2008],
      "model":"Ridge regression fitted to 12 observed prior years",
      "selection_criterion":"minimum validation RMSE; held-out test not used for tuning",
      "alpha_validation_records":tuning,"selected_alpha":chosen,
      "heldout_rmse":rmse,
      "heldout_mae":float(mean_absolute_error(y[test],pred)),
      "persistence_heldout_rmse":base,
      "persistence_heldout_mae":float(mean_absolute_error(y[test],persistence)),
      "test_target_years":ty[test].tolist(),"observed_test":y[test].tolist(),
      "ridge_predictions_test":pred.tolist(),
      "persistence_predictions_test":persistence.tolist()}
    assert np.isclose(rmse,np.sqrt(np.mean((np.array(result["observed_test"])-np.array(result["ridge_predictions_test"]))**2)),rtol=1e-12)
    (out/"figure_02_scientific_discovery_evidence.json").write_text(json.dumps(result,indent=2)+"\n")
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",default=str(Path(__file__).resolve().parents[1]/"figures"))
    args=parser.parse_args()
    r=main(args.output_dir)
    print(json.dumps({k:r[k] for k in
        ["dataset_observations","training_windows","validation_windows",
         "testing_windows","selected_alpha","heldout_rmse","persistence_heldout_rmse"]},indent=2))
