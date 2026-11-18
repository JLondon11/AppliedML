import json, numpy as np, matplotlib.pyplot as plt, statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
D=sm.datasets.nile.load_pandas().data; x=(D.year.to_numpy()-D.year.mean()).reshape(-1,1); y=D.volume.to_numpy(); cut=80
P=PolynomialFeatures(4,include_bias=True); X=P.fit_transform(x); alphas=np.logspace(-4,4,80); errs=[]
for a in alphas:
 m=Ridge(alpha=a,fit_intercept=False).fit(X[:cut],y[:cut]); errs.append(mean_squared_error(y[cut:],m.predict(X[cut:]))**.5)
best=int(np.argmin(errs)); plt.semilogx(alphas,errs); plt.axvline(alphas[best],ls='--'); plt.xlabel('Regularization'); plt.ylabel('Chronological RMSE'); plt.tight_layout(); plt.savefig('result.png',dpi=220); plt.close(); json.dump({'n':len(y),'best_alpha':alphas[best],'rmse':errs[best]},open('metrics.json','w'),indent=2)
