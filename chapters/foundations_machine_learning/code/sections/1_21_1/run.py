import json, numpy as np, matplotlib.pyplot as plt, statsmodels.api as sm
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
D=sm.datasets.stackloss.load_pandas().data; X=D.drop(columns=['STACKLOSS']).to_numpy(); y=D.STACKLOSS.to_numpy(); cv=KFold(7,shuffle=True,random_state=7); p=cross_val_predict(RandomForestRegressor(n_estimators=300,random_state=7),X,y,cv=cv); rmse=mean_squared_error(y,p)**.5
plt.scatter(y,p); lo=min(y.min(),p.min()); hi=max(y.max(),p.max()); plt.plot([lo,hi],[lo,hi],ls='--'); plt.xlabel('Observed stack loss'); plt.ylabel('Out-of-fold prediction'); plt.tight_layout(); plt.savefig('result.png',dpi=220); plt.close(); json.dump({'n':len(y),'rmse':rmse},open('metrics.json','w'),indent=2)
