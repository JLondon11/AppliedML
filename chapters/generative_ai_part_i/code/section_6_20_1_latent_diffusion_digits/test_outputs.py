"""Verify real-data latent diffusion artifacts after run.py (trusted local files only)."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import torch
from sklearn.datasets import load_digits
ROOT=Path(__file__).resolve().parent/'results'
def main():
 m=json.loads((ROOT/'manifest.json').read_text())
 source=load_digits();y=source.target.astype('int64')
 assert hashlib.sha256(source.data.tobytes()+y.tobytes()).hexdigest()==m['source_sha256']
 assert (m['n_train'],m['n_validation'],m['n_test'])==(1257,270,270)
 # The NumPy-array checkpoints are created by run.py; only open trusted own files.
 split=torch.load(ROOT/'split_indices.pt',weights_only=False)
 assert len(set(split['train'])&set(split['test']))==0
 assert len(set(split['train'])&set(split['validation']))==0
 with (ROOT/'predictions.csv').open() as f:rows=list(csv.DictReader(f))
 assert len(rows)==2000
 for arm,key in [('conditional','class_conditional_consistency'),('unconditional','unconditional_consistency')]:
  r=[v for v in rows if v['arm']==arm]
  assert len(r)==1000
  assert {int(x['prompt_label']) for x in r}==set(range(10))
  assert all(abs(sum(float(v[f'p_{k}']) for k in range(10))-1)<1e-5 for v in r)
  consistency=np.mean([int(v['prompt_label'])==int(v['classifier_prediction']) for v in r])
  assert abs(consistency-m[key])<1e-12
  loaded=torch.load(ROOT/f'generated_{arm}.pt',weights_only=False)
  assert loaded['generated'].shape==(1000,64)
  assert loaded['prompts'].shape==(1000,)
 for file in ['digit_conditioned_unconditioned_real.png','prompt_consistency.png']:
  assert (ROOT/'figures'/file).stat().st_size>10000
 print('PASS: real source hash, disjoint 1257/270/270 split, all 2000 predictions, figures')
if __name__=='__main__':main()
