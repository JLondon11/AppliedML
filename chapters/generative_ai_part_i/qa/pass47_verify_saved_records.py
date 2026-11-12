"""Chapter 6 Pass 47: independently recompute genuine-data saved-result integrity.

Expected pass-specific local evidence trees from Chapter6_MasterQA_Scientific_Evidence_Pass47.zip.
This checks genuine source hashes, predictions, checkpoints and metrics.
It does not retrain a model or perform independent physical measurements.
"""
from pathlib import Path
import csv,hashlib,json,sys,subprocess
import numpy as np
import torch
from skimage import data,color,transform
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,roc_auc_score
import statsmodels.api as sm

ROOT=Path(__file__).resolve().parent
# For a cloned repository, provide --evidence-root /path/to/unzipped/archived_evidence
if '--evidence-root' in sys.argv:
    ROOT=Path(sys.argv[sys.argv.index('--evidence-root')+1])

def load(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))
def close(a,b,tol=1e-6):
    assert abs(float(a)-float(b))<tol,(a,b)

# Source photo restoration, 48 held-out metrics and six genuinely trained checkpoints.
p=ROOT/'pass19/Chapter6_Pass19/Restoration_Ablation/results'
m=json.loads((p/'manifest.json').read_text());rows=load(p/'metrics.csv')
assert len(rows)==48 and len(load(p/'training.csv'))==390
for name in m['training_images']+m['heldout_test_images']:
    x=getattr(data,name)()
    if x.ndim==3:x=color.rgb2gray(x[:,:,:3])
    if np.issubdtype(x.dtype,np.integer):
        x=x.astype(np.float32)/np.iinfo(x.dtype).max
    x=transform.resize(np.asarray(x,dtype=np.float32),(128,128),anti_aliasing=True).astype(np.float32)
    assert hashlib.sha256(x.tobytes()).hexdigest()==m['sha256'][name]
for summary in load(p/'summary.csv'):
    arm=[r for r in rows if r['model']==summary['model']]
    assert len(arm)==12
    for metric,key in [('psnr_db','psnr_mean'),('ssim','ssim_mean')]:
        close(np.mean([float(v[metric]) for v in arm]),summary[key],1e-8)
for seed in m['seeds']:
    for arm in ('conditioned','unconditioned'):
        ckpt=torch.load(p/f'model_{arm}_seed{seed}.pt',map_location='cpu',weights_only=True)
        assert ckpt['seed']==seed and ckpt['steps']==65
print('PASS: genuine photos; 48 restoration results; 6 checkpoints')

# True observed CO2 series, 7344 held-out rows, 12 error scores, coverage.
p=ROOT/'pass19/Chapter6_Pass19/Climate_CO2_CVAE/results'
m=json.loads((p/'manifest.json').read_text())
x=sm.datasets.co2.load_pandas().data['co2'].asfreq('W-SAT').to_numpy(dtype=np.float32)
assert hashlib.sha256(np.nan_to_num(x,nan=-999).tobytes()).hexdigest()==m['raw_sha256']
pred=load(p/'test_predictions.csv');rows=load(p/'metrics.csv');coverage=load(p/'interval_coverage.csv')
assert (len(pred),len(rows),len(coverage))==(7344,12,3)
for seed in m['seeds']:
    part=[r for r in pred if int(r['seed'])==seed]
    assert len(part)==2448
    actual=np.array([float(r['true_co2_ppm']) for r in part])
    lo=np.array([float(r['cvae_q05_ppm']) for r in part]);hi=np.array([float(r['cvae_q95_ppm']) for r in part])
    assert np.all(lo<=hi)
    observed_coverage=float(np.mean((actual>=lo)&(actual<=hi)))
    close(observed_coverage,next(r['empirical_coverage'] for r in coverage if int(r['seed'])==seed),1e-9)
    for arm,key in [('CVAE_prior_mean','cvae_mean_ppm'),('ridge','ridge_ppm'),
                    ('persistence','persistence_ppm'),('seasonal_drift','seasonal_drift_ppm')]:
        est=np.array([float(r[key]) for r in part])
        old=next(r for r in rows if int(r['seed'])==seed and r['model']==arm)
        close(np.sqrt(np.mean((actual-est)**2)),old['RMSE_ppm'])
        close(np.mean(np.abs(actual-est)),old['MAE_ppm'])
    ckpt=torch.load(p/f'cvae_seed{seed}.pt',map_location='cpu',weights_only=True)
    assert ckpt['seed']==seed
assert np.mean([float(r['empirical_coverage']) for r in coverage])<.5
print('PASS: genuine CO2 hash; 7344 predictions; 12 scores; undercoverage verified')

# Genuine Wisconsin breast cytology observations, disjoint patient splits, 9 classifiers.
p=ROOT/'pass27/package/clinical_cvae_executed_results'
m=json.loads((p/'manifest.json').read_text())
d=load_breast_cancer();X=d.data.astype(np.float32);y=d.target.astype(np.int64)
assert hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest()==m['data_sha256']
assert (m['train'],m['validation'],m['test'])==(341,114,114)
splits=m['splits']
assert len(set(splits['train_ids'])|set(splits['validation_ids'])|set(splits['test_ids']))==569
assert not set(splits['train_ids'])&set(splits['test_ids'])
assert not set(splits['validation_ids'])&set(splits['test_ids'])
rows=load(p/'metrics.csv');pred=load(p/'test_predictions.csv')
assert (len(rows),len(pred),len(load(p/'prior_samples.csv')))==(9,1026,1023)
for row in rows:
    sub=[r for r in pred if r['seed']==row['seed'] and r['arm']==row['arm']]
    assert len(sub)==114
    actual=np.array([int(r['true_label']) for r in sub]);estimated=np.array([int(r['prediction']) for r in sub])
    probs=np.array([float(r['probability_benign']) for r in sub])
    for metric,val in [('accuracy',accuracy_score(actual,estimated)),
        ('balanced_accuracy',balanced_accuracy_score(actual,estimated)),
        ('macro_F1',f1_score(actual,estimated,average='macro')),
        ('ROC_AUC',roc_auc_score(actual,probs))]:
        close(val,row[metric],1e-9)
for seed in m['seeds']:
    ckpt=torch.load(p/f'cvae_seed{seed}.pt',map_location='cpu',weights_only=False)
    assert ckpt['seed']==seed
print('PASS: genuine cytology hash; 1026 held-out predictions; 9 classifiers')

# Archived standalone evidence checkers; these do not retrain the source models.
checks=[
 ROOT/'pass22/experiments/section_6_medical_histology_vae/test_outputs.py',
 ROOT/'pass23/experiment/test_outputs.py',
 ROOT/'pass20/Molecular_SMILES_VAE/test_outputs.py',
 ROOT/'pass24/experiment/test_outputs.py']
for script in checks:
    result=subprocess.run([sys.executable,str(script)],text=True,capture_output=True)
    if result.returncode:raise RuntimeError(f'{script}: {result.stderr}')
    print(result.stdout.strip())
print('PASS: seven additional genuine-source/output evidence checks; application-level certification remains HOLD')
