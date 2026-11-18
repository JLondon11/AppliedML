import json, numpy as np, matplotlib.pyplot as plt, statsmodels.api as sm, torch
s=sm.datasets.co2.load_pandas().data['co2'].interpolate().bfill().ffill().to_numpy(dtype='float32'); L=52
X=np.stack([s[i:i+L] for i in range(len(s)-L)]); y=np.array([s[i+L] for i in range(len(s)-L)],dtype='float32'); cut=int(.8*len(X)); mu,sd=X[:cut].mean(),X[:cut].std(); X=(X-mu)/sd; yn=(y-mu)/sd
xt=torch.tensor(X[:cut])[:,:,None]; yt=torch.tensor(yn[:cut])[:,None]; xv=torch.tensor(X[cut:])[:,:,None]; yv=torch.tensor(yn[cut:])[:,None]
class T(torch.nn.Module):
 def __init__(self): super().__init__(); self.inp=torch.nn.Linear(1,4); layer=torch.nn.TransformerEncoderLayer(4,1,8,batch_first=True); self.enc=torch.nn.TransformerEncoder(layer,1); self.out=torch.nn.Linear(4,1)
 def forward(self,x): return self.out(self.enc(self.inp(x))[:,-1])
torch.manual_seed(7); m=T(); opt=torch.optim.AdamW(m.parameters(),lr=.01); val=[]
for _ in range(20):
 m.train(); opt.zero_grad(); loss=torch.nn.functional.mse_loss(m(xt),yt); loss.backward(); opt.step(); m.eval()
 with torch.no_grad(): val.append(float(torch.nn.functional.mse_loss(m(xv),yv)))
plt.semilogy(range(1,len(val)+1),val,marker='o'); plt.xlabel('Epoch'); plt.ylabel('Validation MSE (standardized CO2)'); plt.tight_layout(); plt.savefig('result.png',dpi=220); plt.close(); json.dump({'n':len(s),'best_validation_mse':min(val),'final_validation_mse':val[-1]},open('metrics.json','w'),indent=2)
