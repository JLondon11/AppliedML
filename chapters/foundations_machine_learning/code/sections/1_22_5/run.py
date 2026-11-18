import json, numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
X,y=load_breast_cancer(return_X_y=True); Cs=np.logspace(-4,4,24); cv=StratifiedKFold(5,shuffle=True,random_state=7); means=[]; stds=[]
for C in Cs:
 s=cross_val_score(make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=3000)),X,y,cv=cv,scoring='roc_auc'); means.append(s.mean()); stds.append(s.std())
b=int(np.argmax(means)); plt.semilogx(Cs,means); plt.fill_between(Cs,np.array(means)-stds,np.array(means)+stds,alpha=.2); plt.scatter([Cs[b]],[means[b]]); plt.xlabel('C'); plt.ylabel('5-fold ROC AUC'); plt.tight_layout(); plt.savefig('result.png',dpi=220); plt.close(); json.dump({'n':len(y),'best_C':Cs[b],'roc_auc':means[b],'std':stds[b]},open('metrics.json','w'),indent=2)
