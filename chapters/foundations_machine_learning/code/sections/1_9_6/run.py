import json, numpy as np, matplotlib.pyplot as plt, statsmodels.api as sm
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error
fig,axs=plt.subplots(2,2,figsize=(8,6)); metrics={}
X,y=load_digits(return_X_y=True); Cs=np.logspace(-2,1,6); a=[cross_val_score(make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=2000)),X,y,cv=3).mean() for C in Cs]; axs[0,0].semilogx(Cs,a,marker='o'); axs[0,0].set_title('Digits'); axs[0,0].set_ylabel('Accuracy'); metrics['digits_best_accuracy']=max(a)
s=sm.datasets.co2.load_pandas().data.co2.interpolate().bfill().ffill().to_numpy(); L=52; Xc=np.stack([s[i:i+L] for i in range(len(s)-L)]); yc=np.array([s[i+L] for i in range(len(s)-L)]); cut=int(.8*len(Xc)); al=np.logspace(-2,2,6); r=[]
for v in al: p=Ridge(alpha=v).fit(Xc[:cut],yc[:cut]).predict(Xc[cut:]); r.append(mean_squared_error(yc[cut:],p)**.5)
axs[0,1].semilogx(al,r,marker='o'); axs[0,1].set_title('CO2'); axs[0,1].set_ylabel('RMSE (ppm)'); metrics['co2_best_rmse']=min(r)
D=sm.datasets.modechoice.load_pandas().data; ym=D['mode'].astype('category').cat.codes.to_numpy(); Xm=D.drop(columns=['mode']).select_dtypes(include=[np.number]).to_numpy(); am=[cross_val_score(make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=2000)),Xm,ym,cv=3).mean() for C in Cs]; axs[1,0].semilogx(Cs,am,marker='o'); axs[1,0].set_title('ModeChoice'); axs[1,0].set_ylabel('Accuracy'); metrics['modechoice_best_accuracy']=max(am)
N=sm.datasets.nile.load_pandas().data; xn=N.year.to_numpy().reshape(-1,1); yn=N.volume.to_numpy(); cutn=80; an=[]
for v in al: p=Ridge(alpha=v).fit(xn[:cutn],yn[:cutn]).predict(xn[cutn:]); an.append(mean_squared_error(yn[cutn:],p)**.5)
axs[1,1].semilogx(al,an,marker='o'); axs[1,1].set_title('Nile'); axs[1,1].set_ylabel('RMSE'); metrics['nile_best_rmse']=min(an)
for ax in axs.ravel(): ax.set_xlabel('Regularization scale')
plt.tight_layout(); plt.savefig('result.png',dpi=220); plt.close(); json.dump(metrics,open('metrics.json','w'),indent=2)
