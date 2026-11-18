import json, numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
X,y=load_diabetes(return_X_y=True); cv=KFold(10,shuffle=True,random_state=7); alphas=np.logspace(-4,3,60); means=[]; stds=[]
for a in alphas:
 vals=[]
 for tr,te in cv.split(X): vals.append(mean_squared_error(y[te],make_pipeline(StandardScaler(),Ridge(alpha=a)).fit(X[tr],y[tr]).predict(X[te]))**.5)
 means.append(np.mean(vals)); stds.append(np.std(vals))
b=int(np.argmin(means)); plt.semilogx(alphas,means); plt.fill_between(alphas,np.array(means)-stds,np.array(means)+stds,alpha=.2); plt.scatter([alphas[b]],[means[b]]); plt.xlabel('alpha'); plt.ylabel('10-fold RMSE'); plt.tight_layout(); plt.savefig('result.png',dpi=220); plt.close(); json.dump({'n':len(y),'best_alpha':alphas[b],'mean_rmse':means[b]},open('metrics.json','w'),indent=2)
