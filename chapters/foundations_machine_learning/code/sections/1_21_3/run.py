import json, numpy as np, matplotlib.pyplot as plt, statsmodels.api as sm
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score
D=sm.datasets.ccard.load_pandas().data.copy(); y=(D.AVGEXP>=D.AVGEXP.quantile(.9)).astype(int).to_numpy(); X=D.drop(columns=['AVGEXP']).to_numpy(); cv=StratifiedKFold(5,shuffle=True,random_state=7); m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced')); p=cross_val_predict(m,X,y,cv=cv,method='predict_proba')[:,1]; pr,re,_=precision_recall_curve(y,p); ap=average_precision_score(y,p)
plt.plot(re,pr); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.tight_layout(); plt.savefig('result.png',dpi=220); plt.close(); json.dump({'n':len(y),'average_precision':ap,'positive_fraction':float(y.mean())},open('metrics.json','w'),indent=2)
