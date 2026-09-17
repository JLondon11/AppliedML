"""Pass 14 closure for Scientific AI Figures 14 and 25."""
from pathlib import Path
import os,sys,json,shutil,math
import numpy as np,pandas as pd, matplotlib.pyplot as plt
HERE=Path(__file__).resolve();REPO=next(p for p in HERE.parents if (p/'chapters').is_dir())
sys.path.insert(0,str(REPO/'chapters'/'_shared'))
from pass14_closure_utils import copy_repo_asset,save_provenance
ROOT=Path(os.environ.get('PASS14_ARTIFACT_ROOT','pass14_artifact'));CH='11_Scientific_AI';OUT=ROOT/CH/'Figures_Production';OUT.mkdir(parents=True,exist_ok=True);prov={}
# Fig 14: exact repository-retained output from the successful RetinaMNIST training + Grad-CAM experiment.
src=REPO/'chapters'/'scientific_ai'/'ophthalmology'/'figure_14_gradcam_retinal_application'/'figure_14_gradcam_retinal_classification.png'
dst=OUT/'Figure_014_Grad_CAM_explainability_pipeline_for_a_retinal_classifier_showing_grad.png'
prov['14']=copy_repo_asset(src,dst);prov['14'].update({'experiment_script':'chapters/scientific_ai/ophthalmology/figure_14_gradcam_retinal_application/figure_14_gradcam_retinal.py','dataset':'RetinaMNIST / MedMNIST v2','seed':1729,'protocol':'official train/val/test splits; 12-epoch CNN; true Grad-CAM from final convolutional layer'})
# Fig 25: published coNGN Matbench-perovskites fold MAEs plus an executable Bayesian-optimization run on the real Matbench perovskites dataset.
from matminer.datasets import load_dataset
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern,WhiteKernel,ConstantKernel
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
D=load_dataset('matbench_perovskites')
# Robustly locate target and structure columns.
target='e_form' if 'e_form' in D.columns else [c for c in D.columns if D[c].dtype.kind in 'fc'][0]
struct_col='structure' if 'structure' in D.columns else [c for c in D.columns if c!=target][0]
def feat(s):
    comp=s.composition
    zs=[];fr=[]
    for el,amt in sorted(comp.items(),key=lambda kv:kv[0].Z):
        zs.append(float(el.Z));fr.append(float(amt/comp.num_atoms))
    zs=(zs+[0]*4)[:4];fr=(fr+[0]*4)[:4]
    return zs+fr+[float(comp.num_atoms),float(s.volume/comp.num_atoms)]
# Use a fixed subset to keep CPU runtime bounded while remaining a real-data optimization.
rng=np.random.default_rng(1729);idx=rng.choice(len(D),size=min(900,len(D)),replace=False);sub=D.iloc[idx].reset_index(drop=True)
X=np.asarray([feat(s) for s in sub[struct_col]],float);y=sub[target].to_numpy(float)
sc=StandardScaler();Xs=sc.fit_transform(X)
chosen=list(rng.choice(len(sub),size=24,replace=False));remaining=set(range(len(sub)))-set(chosen);acq_hist=[];best_hist=[]
for it in range(24):
    kernel=ConstantKernel(1.0,(1e-2,1e2))*Matern(length_scale=np.ones(Xs.shape[1]),nu=2.5)+WhiteKernel(1e-5)
    gp=GaussianProcessRegressor(kernel=kernel,normalize_y=True,random_state=1729,n_restarts_optimizer=0).fit(Xs[chosen],y[chosen])
    cand=np.array(sorted(remaining));mu,sd=gp.predict(Xs[cand],return_std=True);best=np.min(y[chosen]);imp=best-mu-.005;z=imp/np.maximum(sd,1e-9);ei=imp*norm.cdf(z)+sd*norm.pdf(z);j=int(np.argmax(ei));pick=int(cand[j]);acq_hist.append(float(ei[j]));chosen.append(pick);remaining.remove(pick);best_hist.append(float(np.min(y[chosen])))
# ABX3 idealized perovskite positions.
fig=plt.figure(figsize=(11.4,3.1));ax1=fig.add_subplot(131,projection='3d');
# A corners, B body center, X face centers.
for p in [(i,j,k) for i in [0,1] for j in [0,1] for k in [0,1]]:ax1.scatter(*p,s=35)
ax1.scatter(.5,.5,.5,s=65)
for p in [(.5,.5,0),(.5,.5,1),(.5,0,.5),(.5,1,.5),(0,.5,.5),(1,.5,.5)]:ax1.scatter(*p,s=28)
ax1.set_xticks([]);ax1.set_yticks([]);ax1.set_zticks([]);ax1.set_xlabel('ABX₃',fontsize=9)
ax2=fig.add_subplot(132);folds=np.arange(5);mae=np.array([0.0295,0.0309,0.0277,0.0283,0.0284]);ax2.bar(folds,mae);ax2.set_xticks(folds,[f'fold {i}' for i in folds],rotation=35,ha='right',fontsize=7);ax2.set_ylabel('Published coNGN MAE');ax2.set_ylim(0,.034)
ax3=fig.add_subplot(133);ax3.plot(range(1,len(acq_hist)+1),acq_hist,marker='o',ms=3,label='expected improvement');ax3.set_xlabel('BO iteration');ax3.set_ylabel('Acquisition value');ax3b=ax3.twinx();ax3b.plot(range(1,len(best_hist)+1),best_hist,ls='--',lw=1,label='best observed e_form');ax3b.set_ylabel('Best observed formation energy')
for i,ax in enumerate([ax1,ax2,ax3]):ax.text2D(.5,-.13,f'({chr(97+i)})',transform=ax.transAxes,ha='center') if i==0 else ax.text(.5,-.13,f'({chr(97+i)})',transform=ax.transAxes,ha='center')
fig.tight_layout();p=OUT/'Figure_025_Closed_loop_materials_discovery_pipeline_linking_high_throughput_DFT_c.png';fig.savefig(p,dpi=300,bbox_inches='tight',pad_inches=.03);plt.close(fig)
prov['25']={'published_source':'materialsproject/matbench coNGN full benchmark data','matbench_perovskites_coNGN_fold_mae':mae.tolist(),'real_dataset':'matbench_perovskites','candidate_subset':len(sub),'initial_random_candidates':24,'bo_iterations':24,'seed':1729,'surrogate':'GaussianProcessRegressor Matern 2.5','acquisition':'Expected Improvement','acquisition_values':acq_hist,'best_observed_target':best_hist}
save_provenance(ROOT/CH/'pass14_provenance.json',prov)
