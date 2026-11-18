import json, numpy as np, matplotlib.pyplot as plt, statsmodels.api as sm
D=sm.datasets.nile.load_pandas().data; x=D.year.to_numpy(dtype=float); y=D.volume.to_numpy(dtype=float); z=(x-x.mean())/x.std(); deg=5
X=np.column_stack([z**k for k in range(deg+1)]); zg=np.linspace(z.min(),z.max(),200); G=np.column_stack([zg**k for k in range(deg+1)]); D2=np.diff(np.eye(len(zg)),n=2,axis=0); P=(D2@G).T@(D2@G); lam=10.0
w=np.linalg.solve(X.T@X+lam*P,X.T@y); fit=X@w; rough=float(np.linalg.norm(D2@(G@w))); rmse=float(np.sqrt(np.mean((y-fit)**2)))
plt.scatter(x,y,s=18,label='Observed'); plt.plot(x,fit,label='Structure-regularized fit'); plt.xlabel('Year'); plt.ylabel('Nile flow'); plt.legend(); plt.tight_layout(); plt.savefig('result.png',dpi=220); plt.close(); json.dump({'n':len(y),'lambda':lam,'rmse':rmse,'second_difference_roughness':rough},open('metrics.json','w'),indent=2)
