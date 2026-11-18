import json, numpy as np, matplotlib.pyplot as plt, statsmodels.api as sm
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, RocCurveDisplay
D=sm.datasets.ccard.load_pandas().data.copy(); y=(D.AVGEXP>D.AVGEXP.median()).astype(int).to_numpy(); X=D.drop(columns=['AVGEXP']).to_numpy()
cv=StratifiedKFold(5,shuffle=True,random_state=7); m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)); p=cross_val_predict(m,X,y,cv=cv,method='predict_proba')[:,1]; auc=roc_auc_score(y,p)
RocCurveDisplay.from_predictions(y,p); plt.title('Credit-card expenditure risk screening'); plt.tight_layout(); plt.savefig('result.png',dpi=220); plt.close(); json.dump({'n':len(y),'roc_auc':auc},open('metrics.json','w'),indent=2)
