"""Lightweight validation for the NCI molecular VAE generated artifacts."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
out = ROOT/'results'
metrics = json.loads((out/'metrics.json').read_text())
assert (metrics['n_train'],metrics['n_validation'],metrics['n_test']) == (120,40,40)
assert metrics['source_sha256'] == 'c3eef33eec2c9676a54bbcec6dd1b91a099df9b0d0c8a1b60f5178767e4a3e13'
assert 0 <= metrics['vae_mean_tanimoto'] <= 1
assert 0 <= metrics['pca_mean_tanimoto'] <= 1
with (out/'heldout_predictions.csv').open() as fh:
    rows = list(csv.DictReader(fh))
assert len(rows) == metrics['n_test']
assert len({r['source_row'] for r in rows}) == len(rows)
with (out/'dataset_manifest.csv').open() as fh:
    manifest = list(csv.DictReader(fh))
assert len(manifest) == 200
for split,expected in [('train',120),('validation',40),('test',40)]:
    assert sum(r['split']==split for r in manifest)==expected
assert all((out/'figures'/f).stat().st_size>10_000 for f in [
    'nci_vae_training.png','nci_vae_heldout_benchmark.png','nci_molecular_structure_property_map.png'])
assert (out/'nci_fingerprint_vae.pt').stat().st_size>10_000
print('PASS: manifests, disjoint split assignments, outputs, scientific figures, and checkpoint')
