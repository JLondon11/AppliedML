import json, numpy as np, matplotlib.pyplot as plt, statsmodels.api as sm
D=sm.datasets.nile.load_pandas().data; x=D.year.to_numpy(dtype=float); y=D.volume.to_numpy(dtype=float); z=(x-x.mean())/x.std(); deg=5; cut=80
X=np.column_stack([z**k for k in range(deg+1)]); zg=np.linspace(z.min(),z.max(),200); G=np.column_stack([zg**k for k in range(deg+1)]); D2=np.diff(np.eye(len(zg)),n=2,axis=0); P=(D2@G).T@(D2@G); alphas=np.logspace(-4,4,60); rmses=[]; rough=[]
for a in alphas:
 w=np.linalg.solve(X[:cut].T@X[:cut]+a*P,X[:cut].T@y[:cut]); rmses.append(float(np.sqrt(np.mean((y[cut:]-X[cut:]@w)**2)))); rough.append(float(np.linalg.norm(D2@(G@w))))
b=int(np.argmin(rmses)); plt.loglog(rough,rmses,marker='o',ms=3); plt.scatter([rough[b]],[rmses[b]],s=50); plt.xlabel('Second-difference roughness'); plt.ylabel('Chronological RMSE'); plt.tight_layout(); plt.savefig('result.png',dpi=220); plt.close(); json.dump({'n':len(y),'best_lambda':alphas[b],'validation_rmse':rmses[b],'roughness':rough[b]},open('metrics.json','w'),indent=2)
