"""Validate generated DCGAN evidence after run.py finishes."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent/'results'
manifest=json.loads((ROOT/'manifest.json').read_text())
assert manifest['seed']==20261106
assert set(manifest['source_sha256'])=={'brick','gravel','grass'}
with (ROOT/'metrics.csv').open() as f:rows=list(csv.DictReader(f))
assert len(rows)==9
for texture in ('brick','gravel','grass'):
    subset=[r for r in rows if r['texture']==texture]
    assert len(subset)==3
    assert all(int(r['n_train_patches'])==713 and int(r['n_heldout_patches'])==217 for r in subset)
    generated=np.load(ROOT/f'generated_{texture}.npy')
    assert generated.shape==(100,32,32) and np.isfinite(generated).all()
    state=torch.load(ROOT/f'generator_{texture}.pt',map_location='cpu',weights_only=True)
    assert state['steps']==240
for image in ('gan_texture_samples.png','gan_training_losses.png','gan_texture_feature_comparison.png'):
    assert (ROOT/'figures'/image).stat().st_size>10000
with (ROOT/'training_log.csv').open() as f:assert len(list(csv.DictReader(f)))==720
print('PASS: section 8.8.2 outputs and metadata')
