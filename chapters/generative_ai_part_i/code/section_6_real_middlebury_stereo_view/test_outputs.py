"""Check saved genuine Middlebury CVAE evidence; run after run.py."""
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
from skimage import data
R=Path(__file__).parent/'results'
m=json.loads((R/'manifest.json').read_text())
left,right,disp=data.stereo_motorcycle()
assert hashlib.sha256(left.tobytes()).hexdigest()==m['source_sha256_left']
assert hashlib.sha256(right.tobytes()).hexdigest()==m['source_sha256_right']
assert (m['n_training_patches'],m['n_validation_patches'],m['n_test_patches'])==(855,180,225)
assert m['train_rows'][1]<m['validation_rows'][0]
assert m['validation_rows'][1]<m['test_rows'][0]
q=pd.read_csv(R/'test_metrics.csv');trace=pd.read_csv(R/'training.csv')
assert len(q)==225 and q.test_patch.is_unique
assert len(trace)==22 and trace.step.max()==m['steps']
for col,ref in [('PSNR_copy_left','mean_test_PSNR_copy_left'),('SSIM_copy_left','mean_test_SSIM_copy_left'),
                ('PSNR_CVAE_prior_mean','mean_test_PSNR_CVAE_prior_mean'),('SSIM_CVAE_prior_mean','mean_test_SSIM_CVAE_prior_mean')]:
 assert abs(q[col].mean()-m[ref])<1e-9
assert (q.PSNR_CVAE_prior_mean>q.PSNR_copy_left).sum()==46
assert (q.SSIM_CVAE_prior_mean>q.SSIM_copy_left).sum()==188
# Own locally generated checkpoint contains NumPy scalar metadata; never load untrusted .pt with weights_only=False.
ck=torch.load(R/'conditional_stereo_vae.pt',map_location='cpu',weights_only=False)
assert ck['manifest']['selected_step']==m['selected_step']
for name in ('stereo_real_model_results.pdf','stereo_paired_metrics.pdf'):
 assert (R/'figures'/name).stat().st_size>10000
print('PASS: source hashes, spatial separation, all 225 held-out real-target metrics, trained checkpoint, scientific plots')
