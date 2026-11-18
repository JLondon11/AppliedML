import json, numpy as np, matplotlib.pyplot as plt, statsmodels.api as sm
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
D=sm.datasets.modechoice.load_pandas().data; y=D['mode'].astype('category').cat.codes.to_numpy(); X=D.drop(columns=['mode']).select_dtypes(include=[np.number]).to_numpy(); cv=StratifiedKFold(5,shuffle=True,random_state=7); models={'Logistic':make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000)),'RBF-SVM':make_pipeline(StandardScaler(),SVC()),'RF':RandomForestClassifier(n_estimators=300,random_state=7)}; s={k:cross_val_score(v,X,y,cv=cv).mean() for k,v in models.items()}; plt.barh(list(s),list(s.values())); plt.xlabel('5-fold accuracy'); plt.tight_layout(); plt.savefig('result.png',dpi=220); plt.close(); json.dump({'n':len(y),'accuracy':s},open('metrics.json','w'),indent=2)
