import json, numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
X,y=load_breast_cancer(return_X_y=True); cv=StratifiedKFold(5,shuffle=True,random_state=7); models={'Logistic':make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000)),'RBF-SVM':make_pipeline(StandardScaler(),SVC(probability=True)),'RF':RandomForestClassifier(n_estimators=400,random_state=7),'GB':GradientBoostingClassifier(random_state=7)}; scores={}
for n,m in models.items(): p=cross_val_predict(m,X,y,cv=cv,method='predict_proba')[:,1]; scores[n]=roc_auc_score(y,p)
plt.barh(list(scores),list(scores.values())); plt.xlim(.8,1); plt.xlabel('5-fold ROC AUC'); plt.tight_layout(); plt.savefig('result.png',dpi=220); plt.close(); json.dump({'n':len(y),'roc_auc':scores},open('metrics.json','w'),indent=2)
