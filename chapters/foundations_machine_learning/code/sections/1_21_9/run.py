import json, numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
X,y=load_wine(return_X_y=True); cv=StratifiedKFold(5,shuffle=True,random_state=7); m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000)); p=cross_val_predict(m,X,y,cv=cv,method='predict_proba'); Y=label_binarize(y,classes=np.unique(y)); auc=roc_auc_score(Y,p,average=None); plt.bar(range(len(auc)),auc); plt.xlabel('Wine class'); plt.ylabel('One-vs-rest ROC AUC'); plt.ylim(.8,1); plt.tight_layout(); plt.savefig('result.png',dpi=220); plt.close(); json.dump({'n':len(y),'class_auc':auc.tolist()},open('metrics.json','w'),indent=2)
