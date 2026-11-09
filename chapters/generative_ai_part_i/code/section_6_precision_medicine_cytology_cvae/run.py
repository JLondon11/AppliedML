"""Wisconsin Diagnostic Breast Cancer: conditional VAE data augmentation.
One genuine cytology modality, not multimodal precision medicine.
"""
import csv,json,hashlib,random,argparse
from pathlib import Path
import numpy as np,torch
from torch import nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,f1_score,accuracy_score
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SEEDS=(11,23,37)
class CVAE(nn.Module):
 def __init__(self,d=30,z=10):
  super().__init__()
  self.encoder=nn.Sequential(nn.Linear(d+2,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU())
  self.mu=nn.Linear(64,z);self.lv=nn.Linear(64,z)
  self.decoder=nn.Sequential(nn.Linear(z+2,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,d))
 def prior(self,z,y):return self.decoder(torch.cat((z,nn.functional.one_hot(y,2).float()),1))
 def forward(self,x,y):
  h=self.encoder(torch.cat((x,nn.functional.one_hot(y,2).float()),1));mu=self.mu(h);lv=self.lv(h).clamp(-6,5)
  z=mu+torch.randn_like(mu)*torch.exp(.5*lv)
  return self.prior(z,y),mu,lv
def main():
 a=argparse.ArgumentParser();a.add_argument('--epochs',type=int,default=35)
 a.add_argument('--output',type=Path,default=Path(__file__).parent/'results')
 args=a.parse_args();out=args.output;fig=out/'figures';fig.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
 data=load_breast_cancer();X=data.data.astype(np.float32);Y=data.target.astype(np.int64)
 ids=np.arange(len(Y));train,other=train_test_split(ids,test_size=.4,random_state=20261109,stratify=Y)
 val,test=train_test_split(other,test_size=.5,random_state=20261109,stratify=Y[other])
 assert (len(train),len(val),len(test))==(341,114,114)
 scaler=StandardScaler().fit(X[train]);sx=np.clip(scaler.transform(X),-8,8).astype('float32')
 trainx=sx[train];trainy=Y[train];testx=sx[test];testy=Y[test]
 history=[];metrics=[];predictions=[];samples=[];samples_for_plot={}
 for seed in SEEDS:
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
  model=CVAE();opt=torch.optim.Adam(model.parameters(),lr=.001)
  x=torch.from_numpy(sx);y=torch.from_numpy(Y)
  best=1e9;beststate=None;selected=0
  for epoch in range(1,args.epochs+1):
   model.train();order=torch.tensor(np.random.permutation(train));losses=[]
   for ix in order.split(64):
    rec,mu,lv=model(x[ix],y[ix]);recon=(rec-x[ix]).square().mean();kl=(-.5*(1+lv-mu.square()-lv.exp()).sum(1)).mean()
    loss=recon+.003*kl;opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()))
   model.eval()
   with torch.no_grad():
    va=x[val];yv=y[val];h=model.encoder(torch.cat((va,nn.functional.one_hot(yv,2).float()),1))
    reconstruction=model.prior(model.mu(h),yv);vl=float((reconstruction-va).square().mean())
   history.append((seed,epoch,float(np.mean(losses)),vl))
   if vl<best:best=vl;selected=epoch;beststate={k:v.detach().clone() for k,v in model.state_dict().items()}
  model.load_state_dict(beststate);model.eval()
  rng=np.random.default_rng(seed+1000);generated=[];generated_labels=[]
  with torch.no_grad():
   for label in (0,1):
    n=int((trainy==label).sum())
    gen=model.prior(torch.from_numpy(rng.standard_normal((n,10)).astype('float32')),
      torch.full((n,),label,dtype=torch.long)).numpy()
    generated.append(gen);generated_labels.extend([label]*n)
  generated=np.vstack(generated);generated_labels=np.asarray(generated_labels,dtype=np.int64)
  samples_for_plot[seed]=generated
  for i,(values,label) in enumerate(zip(generated,generated_labels)):
   samples.append([seed,i,int(label),*map(float,values)])
  for arm in ('real_only','real_bootstrap','real_plus_cvae'):
   if arm=='real_only':bx,by=trainx,trainy
   elif arm=='real_bootstrap':
    extra=np.concatenate([rng.choice(np.flatnonzero(trainy==label),size=int((trainy==label).sum()),replace=True) for label in (0,1)])
    bx=np.vstack([trainx,trainx[extra]]);by=np.r_[trainy,trainy[extra]]
   else:bx=np.vstack([trainx,generated]);by=np.r_[trainy,generated_labels]
   clf=LogisticRegression(C=1,max_iter=1000,random_state=seed).fit(bx,by)
   prob=clf.predict_proba(testx)[:,1];cls=(prob>=.5).astype(int)
   metrics.append(dict(seed=seed,arm=arm,n_train_effective=len(bx),accuracy=float(accuracy_score(testy,cls)),
      balanced_accuracy=float(balanced_accuracy_score(testy,cls)),
      macro_F1=float(f1_score(testy,cls,average='macro')),ROC_AUC=float(roc_auc_score(testy,prob))))
   predictions.extend((seed,arm,int(i),int(t),int(p),float(z)) for i,(t,p,z) in enumerate(zip(testy,cls,prob)))
  torch.save({'weights':beststate,'seed':seed,'selected_epoch':selected,
              'scaler_mean':scaler.mean_,'scaler_scale':scaler.scale_},out/f'cvae_seed{seed}.pt')
 with (out/'metrics.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
 with (out/'test_predictions.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['seed','arm','test_index','true_label','prediction','probability_benign']);w.writerows(predictions)
 with (out/'prior_samples.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['seed','index','conditioning_class',*['scaled_'+n.replace(' ','_') for n in data.feature_names]]);w.writerows(samples)
 with (out/'training.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['seed','epoch','train_vae_objective','validation_reconstruction_MSE']);w.writerows(history)
 seed=SEEDS[0];pca=PCA(n_components=2).fit(trainx);real=pca.transform(trainx);generated=pca.transform(samples_for_plot[seed])
 f,ax=plt.subplots(figsize=(7.2,5))
 ax.scatter(real[:,0],real[:,1],s=12,alpha=.32,label='Genuine training cytology')
 ax.scatter(generated[:,0],generated[:,1],s=12,alpha=.32,label='CVAE prior samples')
 ax.set(xlabel='Real-training PCA component 1',ylabel='Real-training PCA component 2')
 ax.legend(frameon=False);f.tight_layout();f.savefig(fig/'real_vs_cvae_cytology.png',dpi=190);plt.close(f)
 f,ax=plt.subplots(figsize=(8,4.2));arms=('real_only','real_bootstrap','real_plus_cvae')
 for j,arm in enumerate(arms):
  vals=np.array([r['ROC_AUC'] for r in metrics if r['arm']==arm])
  ax.scatter(j+np.array([-.1,0,.1]),vals,s=55);ax.hlines(vals.mean(),j-.16,j+.16,lw=2)
 ax.set(xticks=range(3),xticklabels=['Real only','Matched real repeat','Real + CVAE'],ylim=(.7,1.01),ylabel='Held-out ROC AUC')
 ax.grid(axis='y',alpha=.15);f.tight_layout();f.savefig(fig/'real_patient_auc_ablation.png',dpi=190);plt.close(f)
 f,ax=plt.subplots(figsize=(7.5,3.8))
 for s in SEEDS:
  h=[v for v in history if v[0]==s];ax.plot([v[1] for v in h],[v[3] for v in h],label=f'Seed {s}')
 ax.set(xlabel='Training epoch',ylabel='Validation reconstruction MSE')
 ax.legend(frameon=False,fontsize=8);f.tight_layout();f.savefig(fig/'clinical_cvae_validation.png',dpi=190);plt.close(f)
 outmanifest={'dataset':'sklearn Wisconsin Diagnostic Breast Cancer cytology measurements',
  'original_cases':len(Y),'original_features':X.shape[1],
  'data_sha256':hashlib.sha256(X.tobytes()+Y.tobytes()).hexdigest(),
  'train':len(train),'validation':len(val),'test':len(test),
  'seeds':list(SEEDS),'epochs':args.epochs,
  'splits':{'train_ids':train.tolist(),'validation_ids':val.tolist(),'test_ids':test.tolist()},
  'model':'conditional tabular VAE (10 latent dimensions)','n_generated_per_seed':341,
  'scope':'single-modality cytology patient-record features, not multimodal precision medicine',
  'limitations':['Small historical tabular dataset','No independent hospital site or prospective cohort',
  'Not multiple clinical modalities','No treatment response or patient outcome',
  'Synthetic rows not validated for privacy or subgroup fairness',
  'Overlapping test population across three seeds']}
 (out/'manifest.json').write_text(json.dumps(outmanifest,indent=2))
 print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
