import json, numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
X,y=load_digits(return_X_y=True); cv=StratifiedKFold(5,shuffle=True,random_state=7); levels=[16,32,64,256]; acc=[]
for L in levels:
 Xq=np.round(X/16*(L-1))/(L-1)*16; acc.append(cross_val_score(make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000)),Xq,y,cv=cv).mean())
plt.plot(levels,acc,marker='o'); plt.xscale('log',base=2); plt.xlabel('Quantization levels'); plt.ylabel('5-fold accuracy'); plt.tight_layout(); plt.savefig('result.png',dpi=220); plt.close(); json.dump({'n':len(y),'accuracy':dict(zip(map(str,levels),acc))},open('metrics.json','w'),indent=2)
