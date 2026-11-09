"""Verify quality summaries from measured per-seed CSV data and linked checkpoint.

Run: python tools/scientific_qa/verify_quality_evidence.py --results PATH/quality --sampling PATH/ddim_comparison --checkpoint PATH/ddpm/best_ddpm.pt
This verifies evidence arithmetic, not perceptual realism or canonical FID.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

METHODS = {('DDPM', 100), ('DDIM', 100), ('DDIM', 50), ('DDIM', 20)}
FIELDS = (
    'feature_frechet', 'mean_max_digit_probability', 'predicted_class_entropy',
    'predicted_classes_present', 'nearest_train_feature_distance_mean',
    'unusually_close_fraction', 'raw_out_of_range_fraction',
)


def verify(quality: Path, sampling: Path, checkpoint: Path | None = None) -> dict:
    summary = json.loads((quality / 'quality_metrics.json').read_text())
    sampler = json.loads((sampling / 'comparison_metrics.json').read_text())
    with (quality / 'quality_per_seed.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert summary['split'] == {'training': 1257, 'validation': 270, 'independent_test': 270}
    assert (sampler['train'], sampler['validation'], sampler['test']) == (1257, 270, 270)
    assert summary['checkpoint_sha256'] == sampler['checkpoint_sha256']
    if checkpoint is not None:
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == summary['checkpoint_sha256']
    seeds = set(summary['seeds'])
    assert len(seeds) >= 3
    assert len(rows) == len(seeds) * len(METHODS)
    observed = set()
    for r in rows:
        key = (r['method'], int(r['steps']), int(r['seed']))
        assert key[:2] in METHODS and key[2] in seeds
        assert key not in observed, f'duplicate run {key}'
        observed.add(key)
        assert int(r['samples']) == summary['generated_samples_per_seed']
        for field in FIELDS:
            assert math.isfinite(float(r[field])), f'non-finite {key} {field}'
        for field in ('mean_max_digit_probability', 'unusually_close_fraction', 'raw_out_of_range_fraction'):
            assert 0 <= float(r[field]) <= 1
        assert 1 <= float(r['predicted_classes_present']) <= 10
        assert float(r['feature_frechet']) >= 0
    for method, steps in METHODS:
        selected = [r for r in rows if r['method'] == method and int(r['steps']) == steps]
        assert len(selected) == len(seeds)
        reported = summary['metrics'][f'{method}-{steps}']
        for field in FIELDS:
            values = [float(r[field]) for r in selected]
            assert math.isclose(mean(values), reported[field]['mean'], rel_tol=1e-9, abs_tol=1e-9), field
            assert math.isclose(stdev(values), reported[field]['sd_across_seeds'], rel_tol=1e-9, abs_tol=1e-9), field
    assert 0.9 <= summary['independent_test_digit_classification_accuracy'] <= 1
    assert 0 < summary['reference_real_vs_real_feature_frechet_128'] < 100
    return {'runs_verified': len(rows), 'checkpoint_sha256': summary['checkpoint_sha256'],
            'status': 'evidence-arithmetic-verified; perceptual-quality certification pending'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--sampling', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.results, args.sampling, args.checkpoint), indent=2))
