import json, numpy as np, matplotlib.pyplot as plt, statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
D=sm.datasets.modechoice.load_pandas().data.copy(); y=D['mode'].astype('category').cat.codes.to_numpy(); X=StandardScaler().fit_transform(D.drop(columns=['mode']).select_dtypes(include=[np.number]).to_numpy()); Cs=np.logspace(-3,3,20); cv=StratifiedKFold(5,shuffle=True,random_state=7); means=[]
for C in Cs: means.append(cross_val_score(LogisticRegression(C=C,max_iter=2000),X,y,cv=cv,scoring='accuracy').mean())
plt.semilogx(Cs,means,marker='o'); plt.xlabel('Inverse regularization C'); plt.ylabel('5-fold accuracy'); plt.tight_layout(); plt.savefig('result.png',dpi=220); plt.close(); b=int(np.argmax(means)); json.dump({'n':len(y),'best_C':Cs[b],'accuracy':means[b]},open('metrics.json','w'),indent=2)
