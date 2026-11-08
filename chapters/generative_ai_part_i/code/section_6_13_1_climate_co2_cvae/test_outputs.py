"""Recompute full held-out evidence; run after run.py --epochs 65."""
from pathlib import Path
import csv,json
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent/'results'
def read(name):
    with (ROOT/name).open() as file:return list(csv.DictReader(file))

def main():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    assert (manifest['train_windows'],manifest['validation_windows'],manifest['test_windows'])==(1145,254,306)
    assert manifest['epochs']==65
    assert len(manifest['seeds'])==3
    preds=read('test_predictions.csv'); metrics=read('metrics.csv'); coverage=read('interval_coverage.csv')
    assert len(preds)==3*306*8 and len(metrics)==3*4 and len(coverage)==3
    methods={'CVAE_prior_mean':'cvae_mean_ppm','ridge':'ridge_ppm',
             'persistence':'persistence_ppm','seasonal_drift':'seasonal_drift_ppm'}
    for seed in manifest['seeds']:
        subset=[r for r in preds if int(r['seed'])==seed]
        assert len(subset)==306*8
        observed=np.array([float(r['true_co2_ppm']) for r in subset])
        for method,column in methods.items():
            predicted=np.array([float(r[column]) for r in subset])
            rmse=float(np.sqrt(np.mean((predicted-observed)**2)))
            match=next(r for r in metrics if int(r['seed'])==seed and r['model']==method)
            assert abs(rmse-float(match['RMSE_ppm']))<0.0001
        low=np.array([float(r['cvae_q05_ppm']) for r in subset])
        high=np.array([float(r['cvae_q95_ppm']) for r in subset])
        assert np.all(low<=high)
        empirical=float(((observed>=low)&(observed<=high)).mean())
        match=next(r for r in coverage if int(r['seed'])==seed)
        assert abs(empirical-float(match['empirical_coverage']))<1e-8
        checkpoint=torch.load(ROOT/f'cvae_seed{seed}.pt',map_location='cpu',weights_only=True)
        assert checkpoint['seed']==seed
    for filename in ('climate_forecast_benchmark.png','climate_heldout_predictions.png'):
        assert (ROOT/'figures'/filename).stat().st_size>10000
    print('PASS: source split, 3 checkpoints, 7,344 predictions, recomputed RMSE/coverage and figures')
if __name__=='__main__':main()
