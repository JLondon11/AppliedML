"""Real Brownlee stack-loss industrial-process conditional generative benchmark.
Run: python run.py
Requires: numpy pandas scipy sklearn matplotlib statsmodels.
Models and metrics: GP / Ridge / mean on 21 leave-one-out genuine plant values.
"""
import csv,json,hashlib,warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel,Matern,WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error
from statsmodels.datasets import stackloss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent/"results"
ROOT.mkdir(parents=True,exist_ok=True)
real=stackloss.load_pandas().data
X=real[["AIRFLOW","WATERTEMP","ACIDCONC"]].to_numpy(dtype=float)
y=real["STACKLOSS"].to_numpy(dtype=float)
assert X.shape==(21,3)
hashvalue=hashlib.sha256(np.column_stack([y,X]).astype("<f8").tobytes()).hexdigest()
rows=[];samples=[]
for heldout in range(21):
    idx=np.array([j for j in range(21) if j!=heldout])
    scale=StandardScaler().fit(X[idx]);xx=scale.transform(X[idx]);xt=scale.transform(X[heldout:heldout+1])
    kernel=ConstantKernel(1,(.05,30))*Matern(length_scale=np.ones(3),length_scale_bounds=(.15,30),nu=1.5)+WhiteKernel(noise_level=.12,noise_level_bounds=(.003,4))
    gp=GaussianProcessRegressor(kernel=kernel,alpha=1e-6,normalize_y=True,n_restarts_optimizer=0,random_state=20261110)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",UserWarning)
        gp.fit(xx,y[idx]);mean,std=gp.predict(xt,return_std=True)
    mu,sd=float(mean[0]),float(std[0])
    ridge=float(Ridge(alpha=1).fit(xx,y[idx]).predict(xt)[0])
    lower,upper=mu-norm.ppf(.95)*sd,mu+norm.ppf(.95)*sd
    row=dict(heldout_id=heldout+1,air_flow=X[heldout,0],water_temp=X[heldout,1],
       acid_concentration=X[heldout,2],observed_stack_loss=y[heldout],
       gp_predictive_mean=mu,gp_predictive_std=sd,gp_p05=lower,gp_p95=upper,
       ridge_prediction=ridge,training_mean_prediction=float(y[idx].mean()),
       interval_covers_observation=int(lower<=y[heldout]<=upper))
    rows.append(row)
    for j,v in enumerate(np.random.default_rng(20261110+heldout).normal(mu,sd,100)):
        samples.append((heldout+1,j+1,float(v)))
df=pd.DataFrame(rows)
df.to_csv(ROOT/"heldout_predictions.csv",index=False,float_format="%.12g")
with (ROOT/"conditional_generated_samples.csv").open("w",newline="") as f:
    writer=csv.writer(f);writer.writerow(["heldout_id","sample_index","model_generated_stack_loss"]);writer.writerows(samples)
metrics={}
for name,col in [("Gaussian-process emulator","gp_predictive_mean"),("Ridge regression","ridge_prediction"),("Training-mean baseline","training_mean_prediction")]:
    metrics[name]=dict(MAE=float(mean_absolute_error(y,df[col])),RMSE=float(np.sqrt(mean_squared_error(y,df[col]))))
neg=sum(v[2]<0 for v in samples)
manifest=dict(dataset="Brownlee historical ammonia oxidation plant stackloss, statsmodels distribution",
    n_original_observations=21,input_variables=["AIRFLOW","WATERTEMP","ACIDCONC"],target_variable="STACKLOSS",
    source_array_sha256=hashvalue,
    validation="leave one observation out; 20 training and 1 test per fold",
    generative_model="Gaussian-process regression, Matern 3/2 + fitted white noise",
    conditional_samples_per_heldout_condition=100,n_generated_values=len(samples),
    confidence_interval="nominal central 90% Gaussian predictive",
    interval_empirical_coverage=float(df.interval_covers_observation.mean()),
    mean_interval_width=float((df.gp_p95-df.gp_p05).mean()),
    nonnegative_physics_violations_in_generated_draws=int(neg),
    generated_negative_fraction=neg/len(samples),metrics=metrics,
    limitations=["Only 21 plant observations","No independent prospective plant data","Offline static emulator, not closed-loop digital twin","No dynamic physical constraints","Generated values are not observations","Leave-one-out folds have overlapping training data"])
(ROOT/"manifest.json").write_text(json.dumps(manifest,indent=2))
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
navy="#155E75";rust="#C07851";teal="#398273"
fig,ax=plt.subplots(figsize=(8.2,4.5),layout="constrained")
ids=np.arange(1,22)
ax.fill_between(ids,df.gp_p05.to_numpy(),df.gp_p95.to_numpy(),color=navy,alpha=.12,label="GP nominal 90% interval")
ax.plot(ids,df.gp_predictive_mean,"-o",lw=1.4,ms=3,color=navy,label="Held-out GP mean")
ax.scatter(ids,y,marker="D",s=33,color=rust,edgecolors="white",linewidth=.65,label="Measured plant response",zorder=3)
ax.set(xlabel="Plant observation index (not a time-scale estimate)",ylabel="Stack loss (dataset units)",xticks=[1,4,7,10,13,16,19,21])
ax.grid(axis="y",color="#E6ECF0",lw=.75);ax.legend(frameon=False,fontsize=8,ncol=2)
for ext in ("png","pdf"):fig.savefig(ROOT/("stackloss_heldout_predictions."+ext),dpi=220,bbox_inches="tight")
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(9,4.2),layout="constrained")
for name,col,mark,color in [("Gaussian-process emulator","gp_predictive_mean","o",navy),("Ridge regression","ridge_prediction","s",teal),("Training-mean baseline","training_mean_prediction","D",rust)]:
    axes[0].scatter(y,df[col],alpha=.75,s=31,marker=mark,label=name,color=color)
axes[0].plot([y.min()-3,y.max()+3],[y.min()-3,y.max()+3],"--",color="#7A8690",lw=.8)
axes[0].set(xlabel="Observed stack loss",ylabel="Held-out prediction")
axes[0].legend(loc="upper left",frameon=False,fontsize=7.3)
for j,(name,color) in enumerate(zip(metrics,(navy,teal,rust))):
    v=metrics[name]["RMSE"]
    axes[1].plot([j,j],[0,v],lw=2.2,color=color);axes[1].scatter(j,v,s=85,color=color,zorder=3)
    axes[1].text(j,v+.5,f"{v:.2f}",ha="center",fontsize=9)
axes[1].set(xticks=[0,1,2],xticklabels=["GP","Ridge","Mean"],ylabel="Leave-one-out RMSE",
            ylim=(0,max(v["RMSE"] for v in metrics.values())*1.23))
for ax in axes:ax.grid(axis="y",color="#E6ECF0",lw=.75)
for ext in ("png","pdf"):fig.savefig(ROOT/("stackloss_baseline_benchmark."+ext),dpi=220,bbox_inches="tight")
plt.close(fig)
print(json.dumps(manifest,indent=2))
