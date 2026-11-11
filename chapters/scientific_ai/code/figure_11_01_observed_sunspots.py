"""Chapter 11 Figure 11.1: an observation-grounded end-to-end Scientific AI workflow.

Original authentic NGDC annual sunspots via statsmodels. No synthetic data or
inferred model performance. Run python generate_figure_11_01_real.py --out DIR.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from statsmodels.datasets import sunspots

NAVY='#25465E'; TEAL='#2D7A6D'; COPPER='#B8774C'; LIGHT='#9CABAF'; VIOLET='#726B91'

def experiment(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    data=sunspots.load_pandas().data
    years=data.YEAR.to_numpy(dtype=np.int64); values=data.SUNACTIVITY.to_numpy(dtype=np.float64)
    assert len(values)==309 and (years[0],years[-1])==(1700,2008)
    lookback=12
    X=np.array([values[i-lookback:i] for i in range(lookback,len(values))])
    Y=values[lookback:]; target_year=years[lookback:]
    tr=target_year<=1889; val=(target_year>=1890)&(target_year<=1949); test=target_year>=1950
    assert (tr.sum(),val.sum(),test.sum())==(178,60,59)
    mu=X[tr].mean(0); sigma=X[tr].std(0);sigma[sigma==0]=1
    Z=(X-mu)/sigma
    alphas=(.1,1,10,100,1000)
    rows=[]
    for alpha in alphas:
        model=Ridge(alpha=alpha).fit(Z[tr],Y[tr])
        rows.append({'alpha':alpha,'validation_rmse':float(np.sqrt(mean_squared_error(Y[val],model.predict(Z[val]))))})
    best=min(rows,key=lambda d:d['validation_rmse'])['alpha']
    model=Ridge(alpha=best).fit(Z[tr],Y[tr]); prediction=model.predict(Z[test])
    persist=X[test,-1]
    rmse=float(np.sqrt(mean_squared_error(Y[test],prediction)))
    p_rmse=float(np.sqrt(mean_squared_error(Y[test],persist)))
    pca=PCA(n_components=2,svd_solver='full').fit(Z[tr])
    co=pca.transform(Z)
    fig,axes=plt.subplots(2,2,figsize=(12.0,7.35),layout='constrained')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.2,'axes.spines.top':False,'axes.spines.right':False})
    a,b,c,d=axes.flat
    a.plot(years,values,lw=1.12,color=NAVY)
    a.axvspan(1950,2008,color=TEAL,alpha=.09)
    a.set(title='(a) Authentic scientific observations',xlabel='Year',ylabel='Annual sunspot number')
    m=b.imshow(Z[tr,:],aspect='auto',origin='lower',cmap='RdBu_r',vmin=-2.8,vmax=2.8)
    b.set(title='(b) Training-only standardized 12-year windows',xlabel='Past-year lag index',ylabel='Training window index',xticks=[0,2,4,6,8,10,11])
    fig.colorbar(m,ax=b,shrink=.8,label='Standardized measured count')
    c.scatter(co[tr,0],co[tr,1],s=15,alpha=.64,color=TEAL,label=f'Train (n={tr.sum()})',rasterized=True)
    c.scatter(co[val,0],co[val,1],s=15,alpha=.64,color=COPPER,label=f'Validation (n={val.sum()})',rasterized=True)
    c.scatter(co[test,0],co[test,1],s=17,alpha=.72,color=VIOLET,label=f'Test (n={test.sum()})',rasterized=True)
    c.set(title='(c) PCA fitted only on training windows',xlabel='Principal component 1',ylabel='Principal component 2')
    c.legend(frameon=False,fontsize=7.7,loc='upper right')
    d.plot(target_year[test],Y[test],color=NAVY,lw=1.8,label='Recorded target')
    d.plot(target_year[test],prediction,color=COPPER,lw=1.4,label=f'Ridge forecast · RMSE {rmse:.2f}')
    d.plot(target_year[test],persist,color=TEAL,alpha=.53,lw=.95,label=f'Persistence · RMSE {p_rmse:.2f}')
    d.set(title='(d) Measured 1950–2008 held-out evaluation',xlabel='Target year',ylabel='Annual sunspot number')
    d.legend(frameon=False,fontsize=8,loc='upper left')
    for ax in axes.flat:ax.grid(alpha=.11)
    fig.suptitle('From recorded observations to evaluated scientific predictions',fontweight='bold',fontsize=15,color=NAVY)
    for ext in ('png','pdf','svg'):
        fig.savefig(out/f'Figure_11_01_Scientific_AI_Ecosystem.{ext}',dpi=220,facecolor='white',bbox_inches='tight')
    plt.close(fig)
    results={'dataset':'NGDC original annual sunspots distributed via statsmodels','original_records':309,
        'original_years':[1700,2008],'source_float64_sha256':hashlib.sha256(values.tobytes()).hexdigest(),
        'lookback':lookback,'train_windows':int(tr.sum()),'validation_windows':int(val.sum()),'test_windows':int(test.sum()),
        'train_target_years':[1712,1889],'val_target_years':[1890,1949],'test_target_years':[1950,2008],
        'normalization':'fit separately on training windows only','PCA':'fit on training windows only, project all partitions',
        'pca_explained_variance_ratio':pca.explained_variance_ratio_.tolist(),
        'alpha_cv':rows,'chosen_alpha':best,'test_rmse':rmse,'persistence_rmse':p_rmse,
        'test_years':target_year[test].tolist(),'test_actual':Y[test].tolist(),
        'test_predicted':prediction.tolist(),'test_persistence':persist.tolist()}
    assert np.isclose(rmse,np.sqrt(np.mean((np.array(results['test_actual'])-np.array(results['test_predicted']))**2)))
    (out/'Figure_11_01_measured_evidence.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({k:results[k] for k in ('original_records','train_windows','validation_windows','test_windows','chosen_alpha','test_rmse','persistence_rmse')},indent=2))
    return results
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='.')
    experiment(p.parse_args().out)
