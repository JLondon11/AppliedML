import json, numpy as np, matplotlib.pyplot as plt, torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
X,y=load_digits(return_X_y=True); X=X.reshape(-1,1,8,8).astype('float32')/16; Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=7); train=TensorDataset(torch.tensor(Xtr),torch.tensor(ytr)); xv=torch.tensor(Xte); yv=torch.tensor(yte)
class N(torch.nn.Module):
 def __init__(self): super().__init__(); self.f=torch.nn.Sequential(torch.nn.Conv2d(1,16,3,padding=1),torch.nn.ReLU(),torch.nn.MaxPool2d(2),torch.nn.Conv2d(16,32,3,padding=1),torch.nn.ReLU(),torch.nn.AdaptiveAvgPool2d((1,1))); self.h=torch.nn.Linear(32,10)
 def forward(self,x): return self.h(self.f(x).flatten(1))
res={}
for lr in [.001,.01,.05,.1]:
 torch.manual_seed(7); loader=DataLoader(train,batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(7)); m=N(); o=torch.optim.SGD(m.parameters(),lr=lr,momentum=.9)
 for _ in range(15):
  for xb,yb in loader: o.zero_grad(); l=torch.nn.functional.cross_entropy(m(xb),yb); l.backward(); o.step()
 res[str(lr)]=float((m(xv).argmax(1)==yv).float().mean())
plt.semilogx([float(k) for k in res],list(res.values()),marker='o'); plt.xlabel('Learning rate'); plt.ylabel('Validation accuracy'); plt.tight_layout(); plt.savefig('result.png',dpi=220); plt.close(); json.dump({'n':len(y),'accuracy':res},open('metrics.json','w'),indent=2)
