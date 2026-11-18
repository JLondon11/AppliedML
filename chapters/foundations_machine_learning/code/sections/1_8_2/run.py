import json, numpy as np, matplotlib.pyplot as plt, torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
X,y=load_digits(return_X_y=True); X=X.reshape(-1,1,8,8).astype('float32')/16; Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=7)
train=TensorDataset(torch.tensor(Xtr),torch.tensor(ytr)); xv=torch.tensor(Xte); yv=torch.tensor(yte)
class Net(torch.nn.Module):
 def __init__(self): super().__init__(); self.f=torch.nn.Sequential(torch.nn.Conv2d(1,16,3,padding=1),torch.nn.ReLU(),torch.nn.MaxPool2d(2),torch.nn.Conv2d(16,32,3,padding=1),torch.nn.ReLU(),torch.nn.AdaptiveAvgPool2d((1,1))); self.h=torch.nn.Linear(32,10)
 def forward(self,x): return self.h(self.f(x).flatten(1))
curves={}
for name in ['adam','sgd_momentum']:
 torch.manual_seed(7); loader=DataLoader(train,batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(7)); model=Net(); opt=torch.optim.Adam(model.parameters(),lr=.01) if name=='adam' else torch.optim.SGD(model.parameters(),lr=.05,momentum=.9)
 vals=[]
 for ep in range(20):
  model.train()
  for xb,yb in loader: opt.zero_grad(); loss=torch.nn.functional.cross_entropy(model(xb),yb); loss.backward(); opt.step()
  model.eval(); vals.append(float((model(xv).argmax(1)==yv).float().mean()))
 curves[name]=vals
for k,v in curves.items(): plt.plot(range(1,len(v)+1),v,label=k)
plt.xlabel('Epoch'); plt.ylabel('Validation accuracy'); plt.legend(); plt.tight_layout(); plt.savefig('result.png',dpi=220); plt.close(); json.dump({'n':len(y),'final_accuracy':{k:v[-1] for k,v in curves.items()}},open('metrics.json','w'),indent=2)
