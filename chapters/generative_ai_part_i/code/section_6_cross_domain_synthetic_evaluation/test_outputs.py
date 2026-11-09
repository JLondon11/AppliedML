"""Independent recomputation of experimental metrics from saved predictions."""
import csv,json
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
OUT=Path(__file__).resolve().parent/'results'
def load(name):
    with (OUT/name).open(newline='') as f:return list(csv.DictReader(f))
def main():
    meta=json.loads((OUT/'manifest.json').read_text())
    assert meta['source_test']==360 and meta['target_test_tiles']==72
    assert meta['target_train_tiles']==120 and meta['source_train_used']==120
    rows=load('metrics.csv');pred=load('predictions.csv')
    assert len(rows)==6 and len(pred)==2592
    for r in rows:
        records=[x for x in pred if int(x['seed'])==int(r['seed']) and x['arm']==r['arm']]
        assert len(records)==432
        y=[int(x['ground_truth_target_domain']) for x in records]
        p=[float(x['probability_target_domain']) for x in records]
        assert y.count(0)==360 and y.count(1)==72
        assert abs(balanced_accuracy_score(y,[v>=.5 for v in p])-float(r['test_balanced_accuracy']))<1e-10
        assert abs(roc_auc_score(y,p)-float(r['test_roc_auc']))<1e-10
    for figure in ['domain_transfer_accuracy.png','real_neu_domain_predictions.png']:
        assert (OUT/figure).stat().st_size>10000
    print('PASS: 2,592 predictions, six test metrics, and two scientific figures.')
if __name__=='__main__':main()
