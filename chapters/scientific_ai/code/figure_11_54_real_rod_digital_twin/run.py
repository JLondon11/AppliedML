"""Observed-data digital-twin evaluation for Chapter 11 Figure 11.54.

Original dataset: TariqSawaqed1/Heat-Conduction-Aerospace-Lab,
Part 1/Aluminum_26V_250mA.txt (168 timestamps, eight measured temperatures).
Run: python run.py --data path/to/Aluminum_26V_250mA.txt --output-dir outputs
Requires numpy, matplotlib, scikit-learn. No synthetic temperatures or metrics.
"""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

OBS=np.array([0,2,4,6])
HID=np.array([1,3,5,7])
TRAIN_END=100
VAL_END=130
Q_SCALE_GRID=(.03,.1,.3,1.,3.)
R_VAR_GRID=(.005,.03,.1)

def read_log(path):
    lines=Path(path).read_text().splitlines()
    assert lines[0].strip()=="Time[s],CH1[C],CH2[C],CH3[C],CH4[C],CH5[C],CH6[C],CH7[C],CH8[C]"
    raw=np.genfromtxt(path,delimiter=",",skip_header=1)
    if raw.shape!=(168,9) or not np.isfinite(raw).all() or not np.allclose(np.diff(raw[:,0]),10):
        raise ValueError("Expected original 168x9 thermocouple log sampled at ten seconds")
    if raw[0,0]!=0 or raw[-1,0]!=1670:
        raise ValueError("Unexpected time range")
    return raw[:,0],raw[:,1:].astype(np.float64)

def build_model(train):
    prev=train[:-1];target=train[1:]
    model=Ridge(alpha=1.,fit_intercept=True).fit(prev,target)
    A=model.coef_.astype(float);b=model.intercept_.astype(float)
    residual=target-model.predict(prev)
    Q=np.cov(residual,rowvar=False);Q=(Q+Q.T)*.5+np.eye(8)*1e-5
    return A,b,Q

def filtering(temps,A,b,Qbase,qscale,rvar):
    state=temps[TRAIN_END-1].copy()
    cov=np.eye(8)*.02
    H=np.eye(8)[OBS]
    R=np.eye(len(OBS))*rvar
    estimates=[];priors=[]
    for i in range(TRAIN_END,len(temps)):
        xp=A@state+b
        pp=A@cov@A.T+Qbase*qscale
        z=temps[i,OBS]                 # only visible channels at validation/test
        S=H@pp@H.T+R
        K=np.linalg.solve(S,(pp@H.T).T).T
        state=xp+K@(z-H@xp)
        I=np.eye(8)
        cov=(I-K@H)@pp@(I-K@H).T+K@R@K.T
        cov=(cov+cov.T)*.5
        estimates.append(state.copy());priors.append(xp.copy())
    return np.asarray(estimates),np.asarray(priors)

def rmse(actual,pred):return float(np.sqrt(np.mean((actual-pred)**2)))
def mae(actual,pred):return float(np.mean(np.abs(actual-pred)))

def run(data_path,out_dir):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    t,temps=read_log(data_path)
    A,b,Q=build_model(temps[:TRAIN_END])
    static=Ridge(alpha=1.).fit(temps[:TRAIN_END,OBS],
                               temps[:TRAIN_END][:,HID])
    static_pred=static.predict(temps[TRAIN_END:,OBS])
    candidates=[];winner=None
    for qs in Q_SCALE_GRID:
        for rv in R_VAR_GRID:
            filt,_=filtering(temps,A,b,Q,qs,rv)
            err=rmse(temps[TRAIN_END:VAL_END][:,HID],
                     filt[:VAL_END-TRAIN_END][:,HID])
            row={"q_scale":qs,"observation_noise_variance_C2":rv,
                 "val_held_sensor_rmse_C":err}
            candidates.append(row)
            if winner is None or err<winner["val_held_sensor_rmse_C"]:
                winner=row
    filt,prior=filtering(temps,A,b,Q,winner["q_scale"],
                         winner["observation_noise_variance_C2"])
    a=temps[VAL_END:,HID]
    offset=VAL_END-TRAIN_END
    twin=filt[offset:][:,HID]
    static_test=static_pred[offset:]
    prior_test=prior[offset:][:,HID]
    assert a.shape==(38,4) and np.isfinite(twin).all()
    models={"assimilated_twin":twin,"static_sensor_ridge":static_test,
            "one_step_dynamics_prior":prior_test}
    scores={k:{"RMSE_C":rmse(a,v),"MAE_C":mae(a,v)}
            for k,v in models.items()}
    channel_scores={}
    for col,j in enumerate(HID):
        channel_scores[f"CH{j+1}"]={
            "assimilated_rmse_C":rmse(a[:,col],twin[:,col]),
            "static_rmse_C":rmse(a[:,col],static_test[:,col])}
    evidence={
       "source_repository":"https://github.com/TariqSawaqed1/Heat-Conduction-Aerospace-Lab",
       "original_file":"Part 1/Aluminum_26V_250mA.txt",
       "original_source_git_blob_sha":"968990e38e4fe0a1f1b981d117d8bf92a90f163c",
       "local_input_sha256":hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
       "dataset":"Original heated-aluminum thermocouple temperatures, 26V 250mA",
       "recorded_rows":168,"sample_interval_seconds":10,
       "training_times_indices":[0,99],"validation_times_indices":[100,129],
       "test_times_indices":[130,167],
       "training_measurement_count":800,"validation_time_steps":30,
       "heldout_test_time_steps":38,
       "observed_test_channels":[f"CH{i+1}" for i in OBS],
       "withheld_target_channels":[f"CH{i+1}" for i in HID],
       "heldout_target_evaluation_values":152,
       "model":"Ridge empirical 8-state transition with Kalman-style observation assimilation",
       "model_selection":"15 covariance settings selected using 30 validation timestamps only",
       "validation_grid":candidates,"validation_selected":winner,
       "heldout_results":scores,"heldout_per_sensor_results":channel_scores,
       "test_timestamps_seconds":t[VAL_END:].tolist(),
       "test_withheld_measured_C":a.tolist(),
       "test_withheld_assimilated_C":twin.tolist(),
       "test_withheld_static_ridge_C":static_test.tolist(),
       "test_withheld_prior_C":prior_test.tolist(),
       "limitations":["One laboratory rod run with correlated sensor channels",
                      "Empirical state model is not a calibrated PDE",
                      "No calibrated uncertainty or cross-domain performance established"]
    }
    (out/"figure_11_53_measured_evidence.json").write_text(
        json.dumps(evidence,indent=2)+"\n")
    np.savetxt(out/"figure_11_53_measured_sensor_log.csv",
               np.column_stack([t,temps]),delimiter=",",
               header="time_seconds,"+",".join("CH%d_C"%i for i in range(1,9)),comments="")
    palette={"observed":"#183b56","twin":"#2b8078",
             "ridge":"#bd7952","prior":"#786b94"}
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10.2,
                          "axes.spines.top":False,"axes.spines.right":False})
    fig=plt.figure(figsize=(13.4,8.7),layout="constrained")
    gs=fig.add_gridspec(2,2)
    ax0=fig.add_subplot(gs[0,0]);ax1=fig.add_subplot(gs[0,1])
    ax2=fig.add_subplot(gs[1,0]);ax3=fig.add_subplot(gs[1,1])
    for j,col in zip(OBS,["#24455e","#377d75","#b77751","#7a6b91"]):
        ax0.plot(t,temps[:,j],lw=1.5,color=col,label=f"CH{j+1}")
    ax0.axvspan(t[VAL_END],t[-1],color="#dbeeea",alpha=.7)
    ax0.set(title="(a) Original recorded input thermocouples",
            xlabel="Experimental time (s)",ylabel="Temperature (°C)")
    ax0.legend(ncol=4,frameon=False,fontsize=8)
    ch=5;where=list(HID).index(ch)
    ax1.plot(t[VAL_END:],a[:,where],color=palette["observed"],lw=2.4,
             label="Measured CH6")
    ax1.plot(t[VAL_END:],twin[:,where],color=palette["twin"],lw=1.85,
             label="Assimilated estimate")
    ax1.plot(t[VAL_END:],static_test[:,where],color=palette["ridge"],
             lw=1.25,ls="--",label="Static regression")
    ax1.set(title="(b) CH6 held-out comparison (magnified temperature axis)",
            xlabel="Time (s)",ylabel="Temperature (°C)")
    ax1.legend(frameon=False,fontsize=8)
    for label,pred,color in [
        ("Assimilated twin",twin,palette["twin"]),
        ("Static regression",static_test,palette["ridge"]),
        ("Dynamics prior",prior_test,palette["prior"])]:
        err=np.sort(np.abs((a-pred).ravel()))
        ax2.step(err,np.arange(1,len(err)+1)/len(err),where="post",
            lw=2,color=color,label=f"{label} · RMSE {rmse(a,pred):.3f} °C")
    ax2.set(title="(c) Error distribution across 152 held-out measurements",
            xlabel="Absolute sensor reconstruction error (°C)",
            ylabel="Empirical cumulative fraction",ylim=(0,1.03))
    ax2.legend(frameon=False,fontsize=8,loc="lower right")
    for i,j in enumerate(HID):
        ax3.scatter(i,channel_scores[f"CH{j+1}"]["assimilated_rmse_C"],
          marker="o",s=67,color=palette["twin"],
          label="Assimilated twin" if i==0 else None)
        ax3.scatter(i,channel_scores[f"CH{j+1}"]["static_rmse_C"],
          marker="D",s=49,color=palette["ridge"],
          label="Static regression" if i==0 else None)
    ax3.set(xticks=range(4),
        title="(d) Held-out reconstruction by withheld sensor",ylabel="RMSE (°C)")
    ax3.set_xticklabels([f"CH{i+1}" for i in HID])
    ax3.legend(frameon=False,fontsize=9)
    for ax in (ax0,ax1,ax2,ax3):ax.grid(alpha=.13)
    fig.suptitle("Scientific digital twin: real heated-rod measurements and sensor assimilation",
        fontweight="bold",fontsize=15,color=palette["observed"])
    for ext in ("png","pdf","svg"):
        fig.savefig(out/f"Figure_11_53_Bayesian_Multiscale_Digital_Twin_Assimilation.{ext}",
                    dpi=235,bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"validation_selected":winner,"heldout_results":scores,
                      "per_channel":channel_scores},indent=2))
    return evidence

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",required=True)
    parser.add_argument("--output-dir",default="outputs")
    args=parser.parse_args()
    run(args.data,args.output_dir)
