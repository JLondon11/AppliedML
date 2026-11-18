import json, numpy as np, matplotlib.pyplot as plt, statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
D=sm.datasets.modechoice.load_pandas().data.copy(); y=D['mode'].astype('category').cat.codes.to_numpy(); X=D.drop(columns=['mode']).select_dtypes(include=[np.number]).to_numpy(); X=StandardScaler().fit_transform(X); cut=int(.8*len(y)); classes=np.unique(y); m=SGDClassifier(loss='log_loss',alpha=1e-4,random_state=7); acc=[]
for start in range(0,cut,84): end=min(cut,start+84); m.partial_fit(X[start:end],y[start:end],classes=classes); acc.append(accuracy_score(y[cut:],m.predict(X[cut:])))
plt.plot(range(1,len(acc)+1),acc,marker='o'); plt.xlabel('Online update block'); plt.ylabel('Holdout accuracy'); plt.tight_layout(); plt.savefig('result.png',dpi=220); plt.close(); json.dump({'n':len(y),'final_accuracy':acc[-1]},open('metrics.json','w'),indent=2)
