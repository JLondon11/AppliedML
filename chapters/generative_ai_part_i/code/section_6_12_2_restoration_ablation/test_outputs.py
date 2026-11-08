"""Validate complete output from run.py --steps 65."""
import csv,json
from pathlib import Path
import numpy as np
import torch
from skimage import data,color,transform,metrics

OUT=Path(__file__).resolve().parent/'results'
manifest=json.loads((OUT/'manifest.json').read_text())
assert manifest['steps_per_model']==65
assert len(manifest['seeds'])==3
assert set(manifest['training_images']).isdisjoint(manifest['heldout_test_images'])
with (OUT/'metrics.csv').open() as f:rows=list(csv.DictReader(f))
assert len(rows)==3*2*2*4
for r in rows:
    assert np.isfinite(float(r['psnr_db']))
    assert -1<=float(r['ssim'])<=1
for seed in manifest['seeds']:
    for arm in ('conditioned','unconditioned'):
        ckpt=torch.load(OUT/f'model_{arm}_seed{seed}.pt',map_location='cpu',weights_only=True)
        assert ckpt['steps']==65
with (OUT/'training.csv').open() as f:
    trace=list(csv.DictReader(f))
assert len(trace)==3*2*65
for name in ('ablation_psnr.png','real_photo_ablation_outputs.png','ablation_training.png'):
    assert (OUT/'figures'/name).stat().st_size>10000
print('PASS: three seeds x two methods, 48 metrics, all six checkpoints and scientific figures')
