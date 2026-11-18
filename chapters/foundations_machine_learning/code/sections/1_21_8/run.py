import json, numpy as np, matplotlib.pyplot as plt, statsmodels.api as sm
D=sm.datasets.copper.load_pandas().data.copy(); t=D.TIME.to_numpy(dtype=float); Z=[]
for c in [x for x in D.columns if x!='TIME']:
 y=D[c].to_numpy(dtype=float); coef=np.polyfit(t,y,2); resid=y-np.polyval(coef,t); Z.append((resid-resid.mean())/(resid.std() or 1.0))
Z=np.column_stack(Z); score=np.sqrt(np.mean(Z**2,axis=1)); plt.plot(t,score,marker='o'); plt.xlabel('Time'); plt.ylabel('Detrended standardized condition indicator'); plt.tight_layout(); plt.savefig('result.png',dpi=220); plt.close(); json.dump({'n':len(D),'max_indicator':float(score.max()),'mean_indicator':float(score.mean())},open('metrics.json','w'),indent=2)
