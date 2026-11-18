import json, numpy as np, matplotlib.pyplot as plt, statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
D=sm.datasets.copper.load_pandas().data; cols=[c for c in D.columns if c!='TIME']; Z=StandardScaler().fit_transform(D[cols]); score=np.sqrt(np.mean(Z**2,axis=1)); plt.plot(D.TIME,score,marker='o'); plt.xlabel('Time'); plt.ylabel('Standardized condition indicator'); plt.tight_layout(); plt.savefig('result.png',dpi=220); plt.close(); json.dump({'n':len(D),'max_indicator':float(score.max())},open('metrics.json','w'),indent=2)
