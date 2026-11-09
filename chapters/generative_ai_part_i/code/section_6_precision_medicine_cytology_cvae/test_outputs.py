"""Recompute classification metrics from saved held-out predictions."""
import csv,json,hashlib
from pathlib import Path
import numpy as np,torch
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,f1_score,accuracy_score
p=Path(__file__).resolve().parent/'results'
m=json.loads((p/'manifest.json').read_text())
d=load_breast_cancer();X=d.data.astype(np.float32);y=d.target.astype(np.int64)
assert hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest()==m['data_sha256']
assert (m['train'],m['validation'],m['test'])==(341,114,114)
s=m['splits']
assert len(set(s['train_ids'])|set(s['validation_ids'])|set(s['test_ids']))==569
assert not set(s['train_ids'])&set(s['test_ids'])
assert not set(s['validation_ids'])&set(s['test_ids'])
assert not set(s['train_ids'])&set(s['validation_ids'])
with (p/'metrics.csv').open() as f:metrics=list(csv.DictReader(f))
with (p/'test_predictions.csv').open() as f:pred=list(csv.DictReader(f))
with (p/'prior_samples.csv').open() as f:samples=list(csv.DictReader(f))
assert len(metrics)==9 and len(pred)==1026 and len(samples)==1023
assert len({(r['seed'],r['index']) for r in samples})==len(samples)
for r in metrics:
 q=[z for z in pred if z['seed']==r['seed'] and z['arm']==r['arm']]
 assert len(q)==114
 actual=np.array([int(z['true_label']) for z in q])
 proba=np.array([float(z['probability_benign']) for z in q])
 predicted=np.array([int(z['prediction']) for z in q])
 assert 0<=proba.min() and proba.max()<=1
 for col,val in [('ROC_AUC',roc_auc_score(actual,proba)),('accuracy',accuracy_score(actual,predicted)),('balanced_accuracy',balanced_accuracy_score(actual,predicted)),('macro_F1',f1_score(actual,predicted,average='macro'))]:
  assert abs(float(r[col])-val)<1e-10,(r['seed'],r['arm'],col)
for seed in m['seeds']:
 ck=torch.load(p/f'cvae_seed{seed}.pt',map_location='cpu',weights_only=False)
 assert ck['seed']==seed and ck['selected_epoch']<=m['epochs']
for name in ['real_vs_cvae_cytology.png','real_patient_auc_ablation.png','clinical_cvae_validation.png']:
 assert (p/'figures'/name).stat().st_size>15000
print('PASS: genuine data hash, disjoint 341/114/114, 1026 predictions, 1023 synthetic records, all metrics and figures.')
