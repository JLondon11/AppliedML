"""Independently refit the real Brownlee 21-fold leave-one-out GP and two baselines.

No invented plant samples, no claim of plant deployment. Emits original
per-held-out predictions and checks exact source numerical evidence.
"""
import hashlib,json,warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel,Matern,WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error
from statsmodels.datasets import stackloss

def verify(output):
    source=stackloss.load_pandas().data
    X=source[['AIRFLOW','WATERTEMP','ACIDCONC']].to_numpy(dtype=float)
    y=source['STACKLOSS'].to_numpy(dtype=float)
    if X.shape!=(21,3): raise ValueError('Unexpected real Brownlee source dataset')
    records=[]
    for heldout in range(21):
        train=np.array([i for i in range(21) if i!=heldout])
        normer=StandardScaler().fit(X[train])
        xtrain=normer.transform(X[train])
        xtest=normer.transform(X[heldout:heldout+1])
        kernel=ConstantKernel(1,(.05,30))*Matern(
            length_scale=np.ones(3),length_scale_bounds=(.15,30),nu=1.5
        )+WhiteKernel(noise_level=.12,noise_level_bounds=(.003,4))
        gp=GaussianProcessRegressor(kernel=kernel,alpha=1e-6,
            normalize_y=True,n_restarts_optimizer=0,random_state=20261110)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            gp.fit(xtrain,y[train])
            mean,std=gp.predict(xtest,return_std=True)
        mu,sd=float(mean[0]),float(std[0])
        ridge=float(Ridge(alpha=1).fit(xtrain,y[train]).predict(xtest)[0])
        halfwidth=norm.ppf(.95)*sd
        records.append(dict(heldout_id=heldout+1,observed_stack_loss=y[heldout],
            gp_predictive_mean=mu,gp_predictive_std=sd,
            gp_p05=mu-halfwidth,gp_p95=mu+halfwidth,
            ridge_prediction=ridge,
            training_mean_prediction=float(y[train].mean())))
    results=pd.DataFrame(records)
    metrics={key:{'RMSE':float(np.sqrt(mean_squared_error(y,results[col]))),
                  'MAE':float(mean_absolute_error(y,results[col]))}
             for key,col in [('GP','gp_predictive_mean'),('Ridge','ridge_prediction'),
                              ('TrainingMean','training_mean_prediction')]}
    expected={'GP':4.118931861605197,'Ridge':3.647664958998736,
              'TrainingMean':10.422811520890129}
    for key,score in expected.items():
        if abs(metrics[key]['RMSE']-score)>1e-5:
            raise ValueError(f'{key} original out-of-fold RMSE does not reproduce')
    covered=int(((results.gp_p05<=results.observed_stack_loss)&
                 (results.observed_stack_loss<=results.gp_p95)).sum())
    if covered!=19:raise ValueError('Original interval coverage changed')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    results.to_csv(output/'original_brownlee_loo_predictions.csv',index=False)
    report={'dataset':'Brownlee historical industrial stack-loss, 21 authentic observations',
        'source_sha256':hashlib.sha256(np.column_stack([y,X]).astype('<f8').tobytes()).hexdigest(),
        'n_plant_observations':len(y),'held_out_folds':21,'observed_metrics':metrics,
        'nominal_90pct_interval_coverage':f'{covered}/21',
        'mean_nominal_interval_width':float((results.gp_p95-results.gp_p05).mean()),
        'scope':'single original plant historical static emulator; no prospective deployment, dynamic plant twin or physical feasibility certification'}
    (output/'verified_brownlee_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser()
    a.add_argument('--output',type=Path,required=True)
    args=a.parse_args()
    print(json.dumps(verify(args.output),indent=2))
