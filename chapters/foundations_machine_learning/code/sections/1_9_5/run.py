import json, numpy as np, matplotlib.pyplot as plt, statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
D=sm.datasets.modechoice.load_pandas().data.copy(); y=D['mode'].astype('category').cat.codes.to_numpy(); X=D.drop(columns=['mode']).select_dtypes(include=[np.number]).to_numpy(); cut=int(.8*len(y)); scaler=StandardScaler().fit(X[:cut]); X=scaler.transform(X); classes=np.unique(y); alphas=np.logspace(-6,-2,9); acc=[]
for a in alphas:
 m=SGDClassifier(loss='log_loss',alpha=a,random_state=7,learning_rate='optimal')
 for start in range(0,cut,84): m.partial_fit(X[start:min(cut,start+84)],y[start:min(cut,start+84)],classes=classes)
 acc.append(accuracy_score(y[cut:],m.predict(X[cut:])))
b=int(np.argmax(acc)); plt.semilogx(alphas,acc,marker='o'); plt.scatter([alphas[b]],[acc[b]],s=50); plt.xlabel('Online regularization alpha'); plt.ylabel('Logged-choice holdout accuracy'); plt.tight_layout(); plt.savefig('result.png',dpi=220); plt.close(); json.dump({'n':len(y),'best_alpha':alphas[b],'holdout_accuracy':acc[b]},open('metrics.json','w'),indent=2)
