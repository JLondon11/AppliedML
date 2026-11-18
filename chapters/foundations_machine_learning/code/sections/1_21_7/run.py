import json, numpy as np, matplotlib.pyplot as plt, statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
s=sm.datasets.macrodata.load_pandas().data.realcons.to_numpy(); L=8; X=np.stack([s[i:i+L] for i in range(len(s)-L)]); y=np.array([s[i+L] for i in range(len(s)-L)]); c=int(.8*len(X)); m=Ridge(alpha=1).fit(X[:c],y[:c]); p=m.predict(X[c:]); rmse=mean_squared_error(y[c:],p)**.5; plt.plot(y[c:],label='Observed'); plt.plot(p,label='Forecast'); plt.legend(); plt.ylabel('Real consumption'); plt.tight_layout(); plt.savefig('result.png',dpi=220); plt.close(); json.dump({'n':len(s),'rmse':rmse},open('metrics.json','w'),indent=2)
