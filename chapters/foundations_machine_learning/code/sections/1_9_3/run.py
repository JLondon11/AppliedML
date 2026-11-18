import json, numpy as np, matplotlib.pyplot as plt, statsmodels.api as sm, torch
s=sm.datasets.co2.load_pandas().data['co2'].interpolate().bfill().ffill().to_numpy(dtype='float32'); L=52; X=np.stack([s[i:i+L] for i in range(len(s)-L)]); y=np.array([s[i+L] for i in range(len(s)-L)],dtype='float32'); c=int(.8*len(X)); mu,sd=X[:c].mean(),X[:c].std(); xt=torch.tensor(((X[:c]-mu)/sd))[:,:,None]; yt=torch.tensor(((y[:c]-mu)/sd))[:,None]; xv=torch.tensor(((X[c:]-mu)/sd))[:,:,None]; yv=torch.tensor(((y[c:]-mu)/sd))[:,None]
class T(torch.nn.Module):
 def __init__(self): super().__init__(); self.i=torch.nn.Linear(1,4); self.e=torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(4,1,8,batch_first=True),1); self.o=torch.nn.Linear(4,1)
 def forward(self,x): return self.o(self.e(self.i(x))[:,-1])
curves={}
for name in ['adamw','sgd_momentum']:
 torch.manual_seed(7); m=T(); o=torch.optim.AdamW(m.parameters(),lr=.01) if name=='adamw' else torch.optim.SGD(m.parameters(),lr=.03,momentum=.9); vals=[]
 for _ in range(20):
  m.train(); o.zero_grad(); l=torch.nn.functional.mse_loss(m(xt),yt); l.backward(); o.step(); m.eval()
  with torch.no_grad(): vals.append(float(torch.nn.functional.mse_loss(m(xv),yv)))
 curves[name]=vals
for k,v in curves.items(): plt.semilogy(range(1,len(v)+1),v,label=k)
plt.xlabel('Epoch'); plt.ylabel('Validation MSE (standardized CO2)'); plt.legend(); plt.tight_layout(); plt.savefig('result.png',dpi=220); plt.close(); json.dump({'n':len(s),'best_validation_mse':{k:min(v) for k,v in curves.items()}},open('metrics.json','w'),indent=2)
