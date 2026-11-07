"""Analyze full benchmark outputs; report paired seed comparisons descriptively."""
import csv,json
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.metrics import recall_score,f1_score

ROOT=Path(__file__).resolve().parent/'results_multiseed'

def rows(path):
    with path.open() as f:return list(csv.DictReader(f))

def write(name,values):
    with (ROOT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)

def main():
    original=rows(ROOT/'benchmark_metrics.csv')
    extra=ROOT/'repeat_control_metrics.csv'
    if extra.exists():original+=rows(extra)
    seeds=sorted({int(r['seed']) for r in original})
    arms=sorted({r['arm'] for r in original})
    summary=[]
    for arm in arms:
        x=[r for r in original if r['arm']==arm]
        for metric in ('test_accuracy','test_balanced_accuracy','test_macro_F1'):
            a=np.array([float(r[metric]) for r in x])
            summary.append(dict(arm=arm,metric=metric,n_seeds=len(a),
                mean=float(a.mean()),sd=float(a.std(ddof=1)),
                minimum=float(a.min()),maximum=float(a.max())))
    write('summary_by_arm.csv',summary)
    pair=[]
    for arm in arms:
        if arm=='real_only':continue
        differences=np.array([
            float(next(r['test_accuracy'] for r in original if r['arm']==arm and int(r['seed'])==s))-
            float(next(r['test_accuracy'] for r in original if r['arm']=='real_only' and int(r['seed'])==s))
            for s in seeds])
        low,high=stats.t.interval(.95,df=len(seeds)-1,loc=differences.mean(),scale=stats.sem(differences))
        pair.append(dict(arm=arm,reference='real_only',
            mean_accuracy_difference=float(differences.mean()),
            seed_sd=float(differences.std(ddof=1)),
            nominal_t95_lower=float(low),nominal_t95_upper=float(high),
            all_negative=bool((differences<0).all())))
    write('paired_seed_comparison.csv',pair)
    pred=rows(ROOT/'test_predictions.csv')
    if (ROOT/'repeat_control_predictions.csv').exists():
        pred+=rows(ROOT/'repeat_control_predictions.csv')
    classes=[]
    for arm in arms:
        for seed in seeds:
            q=[r for r in pred if r['arm']==arm and int(r['seed'])==seed]
            truth=np.array([int(r['true_class']) for r in q])
            predicted=np.array([int(r['pred_class']) for r in q])
            for cl in range(6):
                classes.append(dict(seed=seed,arm=arm,class_index=cl,
                    recall=float(recall_score(truth,predicted,labels=[cl],average='macro',zero_division=0)),
                    f1=float(f1_score(truth,predicted,labels=[cl],average='macro',zero_division=0))))
    write('per_class_scores.csv',classes)
    (ROOT/'statistical_summary.json').write_text(json.dumps(dict(
        seeds=seeds,summary=summary,pairwise=pair,
        note='Three seeds on same single test set; nominal t intervals are not population confidence bounds.'),
        indent=2))
    print(json.dumps(pair,indent=2))
if __name__=='__main__':main()
