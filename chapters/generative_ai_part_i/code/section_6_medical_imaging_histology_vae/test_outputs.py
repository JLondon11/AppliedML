"""Recompute held-out tissue VAE metrics and validate provenance after run.py."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
import torch
from skimage import data
ROOT=Path(__file__).resolve().parent/'results'
def load(name):
 with (ROOT/name).open(newline='') as f:return list(csv.DictReader(f))
def main():
 m=json.loads((ROOT/'metrics.json').read_text())
 assert hashlib.sha256(data.immunohistochemistry().tobytes()).hexdigest()==m['source_sha256']
 assert (m['n_train_patches'],m['n_val_patches'],m['n_test_patches'])==(589,93,217)
 assert m['n_prior_samples']==128 and m['selected_epoch']<=m['epochs']
 rows=load('heldout_metrics.csv');prior=load('prior_samples.csv');trace=load('training.csv')
 assert len(rows)==217 and len(prior)==128 and len(trace)==m['epochs']
 for col,target in [('recon_PSNR_dB','test_reconstruction_PSNR_dB'),('recon_SSIM','test_reconstruction_SSIM'),('recon_MSE','test_reconstruction_MSE'),('nearest_training_patch_MSE','test_nearest_training_patch_MSE')]:
  assert abs(np.mean([float(r[col]) for r in rows])-m[target])<1e-7
 assert abs(np.mean([float(r['nearest_training_patch_MSE']) for r in prior])-m['prior_nearest_training_patch_MSE'])<1e-7
 assert len({(r['row'],r['col']) for r in rows})==217
 assert all(int(r['row'])>=384 and int(r['row'])+32<=512 for r in rows)
 ck=torch.load(ROOT/'tissue_vae.pt',map_location='cpu',weights_only=True)
 assert ck['selected_epoch']==m['selected_epoch']
 for name in ['tissue_real_recon_generated.png','tissue_vae_training.png']:
  assert (ROOT/'figures'/name).stat().st_size>10000
 print('PASS: genuine tissue hash, separated regions, 217 held-out predictions, 128 prior samples and checkpoint.')
if __name__=='__main__':main()
