"""Chapter 2 comparative case study: gradient flow, plain vs residual deep MLPs.
Real dataset: scikit-learn Digits (NOT MNIST); CPU experiment.
"""
import json, random, platform
from pathlib import Path
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

SEED=2705
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(2)
out=Path(__file__).resolve().parent/'outputs'; out.mkdir(exist_ok=True)
data=load_digits()
X=(data.data / 16.0).astype(np.float32); y=data.target.astype(np.int64)
tr,te=train_test_split(np.arange(len(y)),test_size=0.25,random_state=SEED,stratify=y)
Xt=torch.from_numpy(X[tr]); Yt=torch.from_numpy(y[tr]); Xv=torch.from_numpy(X[te]); Yv=torch.from_numpy(y[te])

class Plain(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Linear(64,64)
        self.layers=nn.ModuleList(nn.Linear(64,64) for _ in range(8))
        self.head=nn.Linear(64,10)
    def forward(self,x):
        x=torch.tanh(self.embed(x))
        for layer in self.layers: x=torch.tanh(layer(x))
        return self.head(x)

class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Linear(64,64)
        self.layers=nn.ModuleList(nn.Linear(64,64) for _ in range(8))
        self.head=nn.Linear(64,10)
    def forward(self,x):
        x=torch.tanh(self.embed(x))
        for layer in self.layers: x=x + 0.25*torch.tanh(layer(x))
        return self.head(x)

models={'plain_tanh':Plain(), 'residual_tanh':Residual()}
criterion=nn.CrossEntropyLoss()
results={}; hist={}; gradients={}
for name,model in models.items():
    optim=torch.optim.Adam(model.parameters(),lr=0.001)
    hist[name]=[]
    optim.zero_grad(); loss=criterion(model(Xt[:128]),Yt[:128]);loss.backward()
    gradients[name]=[float(layer.weight.grad.norm().item()) for layer in model.layers]
    for epoch in range(1,21):
        model.train()
        epoch_gen=torch.Generator().manual_seed(SEED+epoch)
        for batch in torch.randperm(len(Xt),generator=epoch_gen).split(128):
            optim.zero_grad(); loss=criterion(model(Xt[batch]),Yt[batch]);loss.backward();optim.step()
        model.eval()
        with torch.no_grad():
            tr_loss=float(criterion(model(Xt),Yt))
            test_logits=model(Xv); te_loss=float(criterion(test_logits,Yv))
            yp=test_logits.argmax(1).numpy()
            acc=float(accuracy_score(y[te],yp))
        hist[name].append({'epoch':epoch,'train_loss':tr_loss,'test_loss':te_loss,'test_accuracy':acc})
    yp=model(Xv).argmax(1).detach().numpy()
    results[name]={'accuracy':float(accuracy_score(y[te],yp)), 'macro_f1':float(f1_score(y[te],yp,average='macro')),'parameter_count':sum(p.numel() for p in model.parameters()),'confusion_matrix':confusion_matrix(y[te],yp,labels=range(10)).tolist()}

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
for metric,filename,ylabel in [('test_accuracy','accuracy_comparison','Held-out accuracy'),('test_loss','loss_comparison','Held-out cross-entropy')]:
    fig,ax=plt.subplots(figsize=(6.6,3.9))
    for name,style in [('plain_tanh','-'),('residual_tanh','--')]:
        ax.plot([q['epoch'] for q in hist[name]],[q[metric] for q in hist[name]],style,linewidth=2,label=name.replace('_',' '))
    ax.set(xlabel='Training epoch',ylabel=ylabel);ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(out/(filename+'.png'),dpi=220);fig.savefig(out/(filename+'.pdf'));plt.close(fig)
fig,ax=plt.subplots(figsize=(6.6,3.9))
for name,style in [('plain_tanh','-o'),('residual_tanh','--s')]:
    ax.semilogy(range(1,9),gradients[name],style,label=name.replace('_',' '),linewidth=1.6,markersize=4)
ax.set(xlabel='Hidden-layer index (input to output)',ylabel='Weight gradient L2 norm at initialization');ax.grid(alpha=.2);ax.legend(frameon=False)
fig.tight_layout();fig.savefig(out/'gradient_norms.png',dpi=220);fig.savefig(out/'gradient_norms.pdf');plt.close(fig)
for name in models:
    fig,ax=plt.subplots(figsize=(5,4.5)); cm=np.asarray(results[name]['confusion_matrix'])
    im=ax.imshow(cm,cmap='Blues');ax.set(xlabel='Predicted digit',ylabel='True digit',xticks=range(10),yticks=range(10));fig.colorbar(im,ax=ax)
    fig.tight_layout();fig.savefig(out/f'confusion_{name}.png',dpi=220);plt.close(fig)
report={'dataset':'sklearn Digits 8x8, real dataset, NOT MNIST','source':'https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html','seed':SEED,'train_n':len(tr),'test_n':len(te),'epochs':20,'optimizer':'Adam lr=0.001','gradient_measurement':'initialization, first 128 training examples, before parameter updates','plain_vs_residual_note':'Same layer widths, depth and number of parameters; different residual connections, scaling 0.25. Only one fixed split and one seed.','python':platform.python_version(),'torch':torch.__version__,'models':results,'initial_gradient_norms':gradients,'learning_curves':hist}
(out/'metrics.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'metrics':results,'gradients':gradients},indent=2))
