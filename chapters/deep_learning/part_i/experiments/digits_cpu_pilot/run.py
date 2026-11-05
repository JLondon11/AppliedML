"""CPU-only real-data pilot; sklearn Digits 8x8 is NOT MNIST 28x28."""
import json, random, platform
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 17
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(2)
output = Path(__file__).resolve().parent / 'outputs'
output.mkdir(parents=True, exist_ok=True)
D = load_digits()
X = (D.images / 16.0).astype('float32')
y = D.target.astype('int64')
idx = np.arange(len(X))
train_idx, test_idx = train_test_split(idx, test_size=0.25, random_state=SEED, stratify=y)
Xtr=torch.from_numpy(X[train_idx,None]); Ytr=torch.from_numpy(y[train_idx])
Xte=torch.from_numpy(X[test_idx,None]); Yte=torch.from_numpy(y[test_idx])
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(1,12,3,padding=1),nn.ReLU(),nn.Conv2d(12,24,3,padding=1),nn.ReLU(),nn.MaxPool2d(2))
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(24*4*4,48),nn.ReLU(),nn.Linear(48,10))
    def forward(self,x): return self.head(self.features(x))
model=SmallCNN()
optimizer=torch.optim.Adam(model.parameters(),lr=0.002)
criterion=nn.CrossEntropyLoss()
history=[]
for epoch in range(1,13):
    model.train()
    for batch in torch.randperm(len(Xtr)).split(128):
        optimizer.zero_grad()
        loss=criterion(model(Xtr[batch]),Ytr[batch]);loss.backward();optimizer.step()
    model.eval()
    with torch.no_grad():
        train_loss=criterion(model(Xtr),Ytr).item()
        test_loss=criterion(model(Xte),Yte).item()
        yp=model(Xte).argmax(dim=1).numpy()
    acc=accuracy_score(y[test_idx],yp)
    history.append({'epoch':epoch,'train_loss':train_loss,'test_loss':test_loss,'test_accuracy':acc})
with torch.no_grad():
    logits=model(Xte);pred=logits.argmax(1).numpy()
cm=confusion_matrix(y[test_idx],pred,labels=np.arange(10))
metrics={'dataset':'sklearn Digits (8x8 handwritten digits; NOT MNIST)','dataset_samples':len(X),'train_samples':len(Xtr),'test_samples':len(Xte),'split_seed':SEED,'epochs':len(history),'optimizer':'Adam 0.002','model':'two-layer CNN','test_accuracy':float(accuracy_score(y[test_idx],pred)),'test_macro_f1':float(f1_score(y[test_idx],pred,average='macro')),'python':platform.python_version(),'torch':torch.__version__,'sklearn_source':'https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html'}
(output/'metrics.json').write_text(json.dumps({'metrics':metrics,'history':history,'confusion_matrix':cm.tolist()},indent=2))
fig,ax=plt.subplots(figsize=(6.6,3.8))
ax.plot([r['epoch'] for r in history],[r['train_loss'] for r in history],label='Training',lw=2)
ax.plot([r['epoch'] for r in history],[r['test_loss'] for r in history],label='Held-out',lw=2)
ax.set(xlabel='Epoch',ylabel='Cross-entropy loss');ax.legend(frameon=False);ax.grid(alpha=.17)
fig.tight_layout();fig.savefig(output/'training_curves.png',dpi=220);fig.savefig(output/'training_curves.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(5.4,4.6))
im=ax.imshow(cm,cmap='Blues');ax.set(xlabel='Predicted digit',ylabel='True digit',xticks=range(10),yticks=range(10));fig.colorbar(im,ax=ax,label='Count');fig.tight_layout();fig.savefig(output/'confusion_matrix.png',dpi=220);fig.savefig(output/'confusion_matrix.pdf');plt.close(fig)
fig,axes=plt.subplots(3,6,figsize=(10,5.4))
for a,i in zip(axes.flat,range(18)):
    a.imshow(X[test_idx[i]],cmap='gray_r',vmin=0,vmax=1)
    a.set_title(f'T:{y[test_idx[i]]}  P:{pred[i]}',fontsize=9)
    a.axis('off')
fig.tight_layout();fig.savefig(output/'heldout_predictions.png',dpi=220);fig.savefig(output/'heldout_predictions.pdf');plt.close(fig)
with torch.no_grad():
    fm=model.features[0](Xte[0:1]).squeeze(0).numpy()
fig,axes=plt.subplots(3,4,figsize=(7,5.5))
for j,a in enumerate(axes.flat):
    a.imshow(fm[j],cmap='viridis');a.axis('off');a.set_title(f'Channel {j+1}',fontsize=8)
fig.tight_layout();fig.savefig(output/'feature_maps.png',dpi=220);fig.savefig(output/'feature_maps.pdf');plt.close(fig)
torch.save(model.state_dict(), output/'model_weights.pt')
print(json.dumps(metrics,indent=2))
