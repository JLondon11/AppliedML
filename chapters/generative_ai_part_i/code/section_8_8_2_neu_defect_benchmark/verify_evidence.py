"""Recompute genuine NEU held-out metrics and reject incomplete deliverables."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
ROOT=Path(__file__).resolve().parent/'results_multiseed'

def main():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    assert manifest['split_sizes']=={'train':1080,'validation':359,'test':360}
    split=pd.read_csv(ROOT/'dataset_manifest.csv')
    assert len(split)==1799 and split.decoded_sha256.nunique()==1799
    for kind,n in manifest['split_sizes'].items():assert (split.split==kind).sum()==n
    metrics=pd.concat((pd.read_csv(ROOT/'benchmark_metrics.csv'),pd.read_csv(ROOT/'repeat_control_metrics.csv')))
    pred=pd.concat((pd.read_csv(ROOT/'test_predictions.csv'),pd.read_csv(ROOT/'repeat_control_predictions.csv')))
    assert len(metrics)==12 and len(pred)==4320
    for r in metrics.itertuples():
        x=pred[(pred.seed==r.seed)&(pred.arm==r.arm)]
        assert len(x)==360 and x.test_row.nunique()==360
        assert all((x.true_class==k).sum()==60 for k in range(6))
        assert abs(accuracy_score(x.true_class,x.pred_class)-r.test_accuracy)<1e-10
        assert abs(balanced_accuracy_score(x.true_class,x.pred_class)-r.test_balanced_accuracy)<1e-10
        assert abs(f1_score(x.true_class,x.pred_class,average='macro')-r.test_macro_F1)<1e-10
        prob=x[[c for c in x if c.startswith('p_')]].to_numpy()
        assert np.max(abs(prob.sum(axis=1)-1))<1e-5
    for seed in (20261106,20261107,20261108):
        assert (ROOT/f'gan_seed{seed}.pt').stat().st_size>10000
        for arm in metrics.arm.unique():
            assert (ROOT/f'classifier_{arm}_seed{seed}.pt').stat().st_size>10000
    assert len(pd.read_csv(ROOT/'gan_training.csv'))==900
    assert len(pd.read_csv(ROOT/'classifier_training.csv'))==108
    assert len(pd.read_csv(ROOT/'repeat_control_training.csv'))==36
    print('PASS: hashes/splits, complete checkpoints/traces, and 12 recomputed held-out metrics')
if __name__=='__main__':main()
