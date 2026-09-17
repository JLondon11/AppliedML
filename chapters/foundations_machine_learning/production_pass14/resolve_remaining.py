"""Pass 14 closure for Foundations of Machine Learning Figures 10, 25, 28, 29, 31.

All numerical panels use public data or published benchmark values.  Workflow-only
panels contain no invented performance values.
"""
from pathlib import Path
import os, sys, io, json, math, requests
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
HERE=Path(__file__).resolve(); REPO=next(p for p in HERE.parents if (p/'chapters').is_dir())
sys.path.insert(0,str(REPO/'chapters'/'_shared'))
from pass14_closure_utils import save_provenance
ROOT=Path(os.environ.get('PASS14_ARTIFACT_ROOT','pass14_artifact')); CH='05_Foundations_of_Machine_Learning'; OUT=ROOT/CH/'Figures_Production'; OUT.mkdir(parents=True,exist_ok=True)
prov={}; np.random.seed(1729)

def mark(ax,s): ax.text(.5,-.12,s,transform=ax.transAxes,ha='center',va='top',fontsize=9)

# Fig 10: real public sklearn digits, exact PCA/t-SNE/UMAP algorithms.
from umap import UMAP
D=load_digits(); X=D.data; y=D.target
Xp=PCA(n_components=2,random_state=1729).fit_transform(X)
Xt=TSNE(n_components=2,perplexity=30,init='pca',learning_rate='auto',random_state=1729,max_iter=1000).fit_transform(X)
Xu=UMAP(n_components=2,n_neighbors=15,min_dist=.1,random_state=1729).fit_transform(X)
fig,axs=plt.subplots(1,3,figsize=(9.2,3.0))
for i,(ax,Z,name) in enumerate(zip(axs,[Xp,Xt,Xu],['PCA','t-SNE','UMAP'])):
    ax.scatter(Z[:,0],Z[:,1],c=y,cmap='tab10',s=4,alpha=.72,linewidths=0); ax.set_xticks([]);ax.set_yticks([]); ax.set_xlabel(name); mark(ax,f'({chr(97+i)})')
fig.tight_layout(); p=OUT/'Figure_010_PCA_t_SNE_and_UMAP_projections_illustrating_differences_in_variance_pr.png'; fig.savefig(p,dpi=300,bbox_inches='tight',pad_inches=.03);plt.close(fig)
prov['10']={'dataset':'scikit-learn digits','algorithms':['PCA','t-SNE','UMAP'],'seed':1729}

# Fig 25: real Camelyon17-WILDS hospital-domain benchmark/protocol.
# Published WILDS baseline, Table 5: ERM ID-val 93.2±5.2, OOD-val 84.9±3.1, OOD-test 70.3±6.4.
# Public split sizes: train 335,996 from 3 hospitals; OOD validation 34,904; OOD test 85,054.
fig,axs=plt.subplots(1,3,figsize=(10.5,3.1))
axs[0].bar(['Train\n3 hospitals','External val\n1 hospital','External test\n1 hospital'],[335996,34904,85054]);axs[0].set_ylabel('Image patches'); mark(axs[0],'(a)')
acc=[93.2,84.9,70.3]; err=[5.2,3.1,6.4]
axs[1].bar(['ID val','OOD val','OOD test'],acc,yerr=err,capsize=3);axs[1].set_ylim(0,100);axs[1].set_ylabel('ERM accuracy (%)');mark(axs[1],'(b)')
# Gate sequence; no invented performance numbers.
gates=['patient/slide\nsplit','fit +\ncalibrate','ID held-out\nevaluation','external hospital\nvalidation','clinical review','staged\ndeployment','drift\nmonitoring']
axs[2].plot(range(len(gates)),np.ones(len(gates)),marker='o',lw=1.2);axs[2].set_yticks([]);axs[2].set_xticks(range(len(gates)),gates,rotation=48,ha='right',fontsize=7);axs[2].set_ylim(.7,1.3);mark(axs[2],'(c)')
for ax in axs: ax.grid(False)
fig.tight_layout();p=OUT/'Figure_025_Medical_image_classification_benchmark_and_deployment_workflow.png';fig.savefig(p,dpi=300,bbox_inches='tight',pad_inches=.03);plt.close(fig)
prov['25']={'dataset':'Camelyon17-WILDS','source':'WILDS paper Table 5 and public split metadata','split_sizes':[335996,34904,85054],'ERM_accuracy_mean':[93.2,84.9,70.3],'ERM_accuracy_sd':[5.2,3.1,6.4]}

# Fig 28: WeatherBench/WeatherBench2 protocol with published baseline scores; no synthetic skill values.
# Numerical baseline values are from Rasp et al. 2020 Table 2; protocol panel follows WB2 evaluation.
models=['Persistence','Linear reg.','CNN direct','IFS T63','Operational IFS']
z3=[936,693,626,268,154]; z5=[1033,783,757,463,334]
t3=[4.23,3.19,2.87,1.85,1.36]; t5=[4.56,3.44,3.37,2.52,2.03]
fig,axs=plt.subplots(1,3,figsize=(11.2,3.2))
x=np.arange(len(models));w=.35
axs[0].bar(x-w/2,z3,w,label='3 d');axs[0].bar(x+w/2,z5,w,label='5 d');axs[0].set_xticks(x,models,rotation=35,ha='right',fontsize=7);axs[0].set_ylabel('Z500 RMSE (m² s⁻²)');axs[0].legend(frameon=False,fontsize=7);mark(axs[0],'(a)')
axs[1].bar(x-w/2,t3,w,label='3 d');axs[1].bar(x+w/2,t5,w,label='5 d');axs[1].set_xticks(x,models,rotation=35,ha='right',fontsize=7);axs[1].set_ylabel('T850 RMSE (K)');mark(axs[1],'(b)')
wb2=['ERA5\n0.25°','regrid','autoregressive\nrollout','RMSE / ACC\n/ CRPS','physical\ndiagnostics','extreme-event\nstrata']
axs[2].plot(range(len(wb2)),np.ones(len(wb2)),marker='o',lw=1.2);axs[2].set_yticks([]);axs[2].set_xticks(range(len(wb2)),wb2,rotation=45,ha='right',fontsize=7);axs[2].set_ylim(.7,1.3);mark(axs[2],'(c)')
for ax in axs: ax.grid(False)
fig.tight_layout();p=OUT/'Figure_028_Scientific_machine_learning_workflow_for_weather_forecasting.png';fig.savefig(p,dpi=300,bbox_inches='tight',pad_inches=.03);plt.close(fig)
prov['28']={'protocol':'WeatherBench 2 evaluation semantics; ERA5 ground truth; RMSE/ACC/CRPS','published_baseline_source':'Rasp et al. 2020 WeatherBench Table 2','Z500_RMSE_3d':dict(zip(models,z3)),'Z500_RMSE_5d':dict(zip(models,z5)),'T850_RMSE_3d':dict(zip(models,t3)),'T850_RMSE_5d':dict(zip(models,t5))}

# Fig 29: MLPerf Tiny quality targets + public measured edge result.  The figure distinguishes benchmark
# constraints from a measured 2026 VWW energy example (ASYGN ColibriNPU, 22.2 uJ/inference).
tasks=['KWS','VWW','Image cls.','Anomaly det.']; quality=[90,80,85,85]; qlabel=['Top-1 %','Top-1 %','Top-1 %','AUC ×100']
fig,axs=plt.subplots(1,3,figsize=(10.3,3.0))
axs[0].bar(tasks,quality);axs[0].set_ylim(0,100);axs[0].set_ylabel('MLPerf Tiny quality target');axs[0].tick_params(axis='x',rotation=25);mark(axs[0],'(a)')
axs[1].bar(['ColibriNPU\nVWW'],[22.2]);axs[1].set_ylabel('Measured energy (µJ / inference)');mark(axs[1],'(b)')
edge=['sensor\ninput','on-device\npreprocess','quantized\ninference','quality\ngate','latency / energy\nmeasurement','firmware\nstage + rollback']
axs[2].plot(range(len(edge)),np.ones(len(edge)),marker='o',lw=1.2);axs[2].set_yticks([]);axs[2].set_xticks(range(len(edge)),edge,rotation=45,ha='right',fontsize=7);axs[2].set_ylim(.7,1.3);mark(axs[2],'(c)')
for ax in axs: ax.grid(False)
fig.tight_layout();p=OUT/'Figure_029_Edge_AI_benchmark_and_deployment_workflow.png';fig.savefig(p,dpi=300,bbox_inches='tight',pad_inches=.03);plt.close(fig)
prov['29']={'benchmark':'MLPerf Tiny','quality_targets':dict(zip(tasks,quality)),'measured_example':{'system':'ASYGN ColibriNPU','task':'Visual Wake Words','energy_uJ_per_inference':22.2,'source':'MLCommons Tiny v1.4 results article, 2026-07-07'}}

# Fig 31: actual C-MAPSS FD001 engine-level split + RF RUL model and PHM08 score.
url='https://raw.githubusercontent.com/mapr-demos/predictive-maintenance/master/notebooks/jupyter/Dataset/CMAPSSData/train_FD001.txt'
r=requests.get(url,timeout=90);r.raise_for_status();raw=pd.read_csv(io.BytesIO(r.content),sep=r'\s+',header=None)
raw=raw.iloc[:,:26]; raw.columns=['unit','cycle','op1','op2','op3']+[f's{i}' for i in range(1,22)]
last=raw.groupby('unit').cycle.max(); raw['RUL']=raw.apply(lambda z:last.loc[z.unit]-z.cycle,axis=1)
units=np.array(sorted(raw.unit.unique()));rng=np.random.default_rng(1729);rng.shuffle(units);tr_u=units[:70];te_u=units[70:]
features=['cycle']+[f's{i}' for i in range(1,22)]
tr=raw[raw.unit.isin(tr_u)];te=raw[raw.unit.isin(te_u)]
rf=RandomForestRegressor(n_estimators=120,min_samples_leaf=3,random_state=1729,n_jobs=-1);rf.fit(tr[features],tr.RUL);pred=rf.predict(te[features]);rmse=float(mean_squared_error(te.RUL,pred)**.5)
def phm08(y,p):
    d=p-y; return float(np.sum(np.where(d<0,np.exp(-d/13)-1,np.exp(d/10)-1)))
score=phm08(te.RUL.to_numpy(),pred)
# one held-out engine trajectory
uid=int(te_u[0]);one=te[te.unit==uid];op=rf.predict(one[features])
fig,axs=plt.subplots(1,3,figsize=(10.4,3.0))
axs[0].scatter(te.RUL,pred,s=4,alpha=.25);lim=max(te.RUL.max(),pred.max());axs[0].plot([0,lim],[0,lim],ls='--',lw=1);axs[0].set_xlabel('True RUL (cycles)');axs[0].set_ylabel('Predicted RUL');mark(axs[0],'(a)')
axs[1].plot(one.cycle,one.RUL,label='true');axs[1].plot(one.cycle,op,label='RF prediction');axs[1].set_xlabel('Cycle');axs[1].set_ylabel('RUL');axs[1].legend(frameon=False,fontsize=7);mark(axs[1],'(b)')
axs[2].bar(['RMSE','PHM08 / 100'],[rmse,score/100]);axs[2].set_ylabel('Evaluation value');mark(axs[2],'(c)')
for ax in axs:ax.grid(False)
fig.tight_layout();p=OUT/'Figure_031_Aerospace_prognostics_workflow_for_NASA_C_MAPSS.png';fig.savefig(p,dpi=300,bbox_inches='tight',pad_inches=.03);plt.close(fig)
prov['31']={'dataset':'NASA C-MAPSS FD001 public mirror','url':url,'split':'70 train engines / 30 held-out engines, seed 1729','model':'RandomForestRegressor 120 trees','rmse':rmse,'phm08_score':score,'note':'RUL constructed from each engine run-to-failure trajectory; split is engine-disjoint.'}
save_provenance(ROOT/CH/'pass14_provenance.json',prov)
