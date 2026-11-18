import json, numpy as np, matplotlib.pyplot as plt, statsmodels.api as sm
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
D=sm.datasets.nile.load_pandas().data; X=D.year.to_numpy().reshape(-1,1); y=D.volume.to_numpy(); cut=80; alphas=np.logspace(-4,4,50); tr=[]; va=[]
for a in alphas:
 m=make_pipeline(SplineTransformer(n_knots=8,degree=3),StandardScaler(with_mean=False),Ridge(alpha=a)).fit(X[:cut],y[:cut]); tr.append(mean_squared_error(y[:cut],m.predict(X[:cut]))**.5); va.append(mean_squared_error(y[cut:],m.predict(X[cut:]))**.5)
plt.semilogx(alphas,tr,label='Train'); plt.semilogx(alphas,va,label='Validation'); plt.legend(); plt.xlabel('alpha'); plt.ylabel('RMSE'); plt.tight_layout(); plt.savefig('result.png',dpi=220); plt.close(); b=int(np.argmin(va)); json.dump({'n':len(y),'best_alpha':alphas[b],'validation_rmse':va[b]},open('metrics.json','w'),indent=2)
