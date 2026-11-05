"""Real-data gradient-flow case study: scikit-learn Digits 8x8, NOT MNIST.
Three seeds, common disjoint train/validation/test split, CPU-compatible.
"""
from pathlib import Path
import json, random, platform
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

OUT=Path(__file__).resolve().parent/'outputs'; OUT.mkdir(exist_ok=True)
torch.set_num_threads(2)
SPLIT_SEED=20261105; SEEDS=(21,42,84); EPOCHS=35
D=load_digits(); X=torch.tensor((D.data / 16.).astype('float32')); y=D.target.astype('int64')
indices=np.arange(len(y))
tr, remainder=train_test_split(indices,test_size=.4,random_state=SPLIT_SEED,stratify=y)
va, te=train_test_split(remainder,test_size=.5,random_state=SPLIT_SEED+1,stratify=y[remainder])
assert len(tr)==1078 and len(va)==359 and len(te)==360
assert len(set(tr)&set(va))==len(set(tr)&set(te))==len(set(va)&set(te))==0
Xt,Yt=X[tr],torch.tensor(y[tr]); Xv,Yv=X[va],torch.tensor(y[va]); Xe,Ye=X[te],torch.tensor(y[te])

class DeepMLP(nn.Module):
 def __init__(self,residual=False):
  super().__init__();self.residual=residual
  self.embed=nn.Linear(64,64)
  self.layers=nn.ModuleList(nn.Linear(64,64) for _ in range(8))
  self.head=nn.Linear(64,10)
 def forward(self,x):
  x=torch.tanh(self.embed(x))
  for layer in self.layers:
   h=torch.tanh(layer(x));x=x+0.25*h if self.residual else h
  return self.head(x)

all_results=[]; curves={}; grads={}; selected_cms={}
for seed in SEEDS:
 for kind in ('plain','residual'):
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
  model=DeepMLP(residual=(kind=='residual'))
  optimizer=torch.optim.Adam(model.parameters(),lr=.001)
  lossfn=nn.CrossEntropyLoss()
  model.zero_grad(set_to_none=True)
  lossfn(model(Xt[:128]),Yt[:128]).backward()
  initial_grad=[float(layer.weight.grad.norm().item()) for layer in model.layers]
  grads[f'{kind}_{seed}']=initial_grad
  best_validation=float('inf'); best_state=None; best_epoch=0; history=[]
  for epoch in range(1,EPOCHS+1):
   model.train();generator=torch.Generator().manual_seed(seed*1000+epoch)
   for batch in torch.randperm(len(tr),generator=generator).split(128):
    optimizer.zero_grad(set_to_none=True)
    lossfn(model(Xt[batch]),Yt[batch]).backward();optimizer.step()
   model.eval()
   with torch.no_grad():
    tl=float(lossfn(model(Xt),Yt));vl=float(lossfn(model(Xv),Yv))
    vacc=float(accuracy_score(y[va],model(Xv).argmax(1).numpy()))
   history.append(dict(epoch=epoch,train_loss=tl,validation_loss=vl,validation_accuracy=vacc))
   if vl<best_validation:
    best_validation=vl;best_epoch=epoch
    best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
  model.load_state_dict(best_state);model.eval()
  with torch.no_grad():
   pred=model(Xe).argmax(1).numpy()
   test_loss=float(lossfn(model(Xe),Ye))
  cm=confusion_matrix(y[te],pred,labels=range(10))
  if seed==SEEDS[0]: selected_cms[kind]=cm.tolist()
  record={'seed':seed,'architecture':kind,'train_samples':len(tr),'validation_samples':len(va),'test_samples':len(te),
   'best_validation_epoch':best_epoch,'best_validation_loss':best_validation,
   'test_accuracy':float(accuracy_score(y[te],pred)),
   'test_macro_f1':float(f1_score(y[te],pred,average='macro')),
   'test_cross_entropy':test_loss,'correct_test_predictions':int(np.sum(pred==y[te])),
   'parameters':sum(p.numel() for p in model.parameters())}
  all_results.append(record);curves[f'{kind}_{seed}']=history
  print(record,flush=True)

summaries={}
for kind in ('plain','residual'):
 rows=[q for q in all_results if q['architecture']==kind]
 summaries[kind]={m:{'mean':float(np.mean([r[m] for r in rows])),
                         'std_sample':float(np.std([r[m] for r in rows],ddof=1))}
                  for m in ['test_accuracy','test_macro_f1','test_cross_entropy']}
paired={str(seed):next(r['test_accuracy'] for r in all_results if r['seed']==seed and r['architecture']=='residual')-next(r['test_accuracy'] for r in all_results if r['seed']==seed and r['architecture']=='plain') for seed in SEEDS}
output={'dataset':'scikit-learn Digits (8x8 handwritten images), NOT MNIST',
 'dataset_url':'https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html',
 'total_samples':len(y),'training_samples':len(tr),'validation_samples':len(va),'testing_samples':len(te),
 'split_seed':SPLIT_SEED,'seeds':list(SEEDS),'epochs_per_run':EPOCHS,
 'model':'8-layer 64-width tanh MLP, with or without scaled residual connections (0.25)',
 'model_selection':'Minimum validation cross-entropy checkpoint, independently per seed and architecture',
 'optimizer':'Adam learning rate 0.001','results':all_results,'aggregate':summaries,
 'paired_accuracy_improvements':paired,'curves':curves,'initial_gradient_norms':grads,
 'confusion_matrices_seed21':selected_cms,
 'limitations':'Three initializations, one fixed dataset partition; no statistical generalization claim or MNIST benchmark.',
 'software':{'python':platform.python_version(),'torch':torch.__version__}}
(OUT/'results.json').write_text(json.dumps(output,indent=2))

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(7,4.1))
for kind,marker in [('plain','o'),('residual','s')]:
 rows=[r for r in all_results if r['architecture']==kind]
 ax.plot([i+1 for i in range(len(rows))],[r['test_accuracy'] for r in rows],
         marker=marker,linewidth=1.8,markersize=7,label=kind.capitalize())
ax.set(xlabel='Initialization seed index (21, 42, 84)',ylabel='Held-out test accuracy',xticks=[1,2,3],ylim=(0.65,1.02))
ax.legend(frameon=False);ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(OUT/'test_accuracy_seeds.png',dpi=220);fig.savefig(OUT/'test_accuracy_seeds.pdf');plt.close(fig)

fig,ax=plt.subplots(figsize=(7,4.1))
for kind,sty in [('plain','-'),('residual','--')]:
 curves_kind=np.array([[q['validation_loss'] for q in curves[f'{kind}_{seed}']] for seed in SEEDS])
 mean=curves_kind.mean(axis=0);sd=curves_kind.std(axis=0,ddof=1)
 xx=np.arange(1,EPOCHS+1)
 ax.plot(xx,mean,sty,linewidth=2,label=kind.capitalize())
 ax.fill_between(xx,mean-sd,mean+sd,alpha=.16)
ax.set(xlabel='Epoch',ylabel='Validation cross-entropy',yscale='log')
ax.grid(alpha=.2);ax.legend(frameon=False)
fig.tight_layout();fig.savefig(OUT/'validation_loss_seeds.png',dpi=220);fig.savefig(OUT/'validation_loss_seeds.pdf');plt.close(fig)

fig,ax=plt.subplots(figsize=(7,4.1))
for kind,sty in [('plain','-o'),('residual','--s')]:
 gv=np.array([grads[f'{kind}_{seed}'] for seed in SEEDS]);avg=gv.mean(axis=0)
 ax.semilogy(range(1,9),avg,sty,linewidth=1.8,markersize=5,label=kind.capitalize())
ax.set(xlabel='Layer index (input to output)',ylabel='Mean initial weight-gradient L2 norm')
ax.grid(alpha=.2);ax.legend(frameon=False)
fig.tight_layout();fig.savefig(OUT/'gradient_norms_seeds.png',dpi=220);fig.savefig(OUT/'gradient_norms_seeds.pdf');plt.close(fig)
print('AGGREGATES',json.dumps(summaries,indent=2));print('PAIRED',paired)
