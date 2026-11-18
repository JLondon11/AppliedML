import json, numpy as np, matplotlib.pyplot as plt, statsmodels.api as sm
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
metrics={}
X,y=load_digits(return_X_y=True); metrics['Digits accuracy']=cross_val_score(make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)),X,y,cv=3).mean()
co2=sm.datasets.co2.load_pandas().data.co2.interpolate().bfill().ffill().to_numpy(); pred=np.repeat(np.mean(co2[:-52]),52); metrics['CO2 normalized score']=1/(1+np.sqrt(np.mean((co2[-52:]-pred)**2)))
D=sm.datasets.modechoice.load_pandas().data; metrics['ModeChoice majority accuracy']=D['mode'].value_counts(normalize=True).iloc[0]
nile=sm.datasets.nile.load_pandas().data.volume.to_numpy(); metrics['Nile normalized score']=1/(1+np.std(nile)/1000)
plt.barh(list(metrics),list(metrics.values())); plt.xlabel('Native normalized diagnostic score'); plt.tight_layout(); plt.savefig('result.png',dpi=220); plt.close(); json.dump(metrics,open('metrics.json','w'),indent=2)
