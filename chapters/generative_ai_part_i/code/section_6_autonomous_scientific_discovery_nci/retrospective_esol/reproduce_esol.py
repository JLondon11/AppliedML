"""Retrospective hit discovery validated against published measured ESOL log-solubility.

Run: python reproduce_esol.py --csv delaney-processed.csv
Data: deepchem/deepchem datasets/delaney-processed.csv; Delaney (2004).
This is NOT a new wet-lab assay, or validation of NCI generated molecules.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
F=['Molecular Weight','Number of H-Bond Donors','Number of Rings','Number of Rotatable Bonds','Polar Surface Area']
Y='measured log solubility in mols per litre'
P='ESOL predicted log solubility in mols per litre'
def shuffle(records,seed=20261109):
    state=seed&0xffffffff; a=list(records)
    for i in range(len(a)-1,0,-1):
        state^=(state<<13)&0xffffffff;state^=state>>17
        state^=(state<<5)&0xffffffff;state&=0xffffffff
        j=(state*(i+1))//4294967296;a[i],a[j]=a[j],a[i]
    return a
def main():
    p=argparse.ArgumentParser();p.add_argument('--csv',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'results')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with args.csv.open(newline='') as fh:original=list(csv.DictReader(fh))
    source=[];seen=set()
    for r in original:
        try:x=np.array([float(r[c]) for c in F]);y=float(r[Y]);v=float(r[P])
        except (ValueError,KeyError):continue
        if not np.all(np.isfinite([*x,y,v])) or r['smiles'] in seen:continue
        seen.add(r['smiles']);source.append(dict(name=r['Compound ID'],smiles=r['smiles'],
                                               features=x,y=y,p=v))
    assert len(source)==1128
    ordered=shuffle(source);train,test=ordered[:902],ordered[902:]
    X=np.stack([r['features'] for r in train]);y=np.array([r['y'] for r in train])
    mean=X.mean(0);sd=X.std(0)
    design=np.column_stack((np.ones(len(X)),(X-mean)/sd))
    penalty=np.eye(6)*10;penalty[0,0]=0
    beta=np.linalg.solve(design.T@design+penalty,design.T@y)
    sample=np.stack([r['features'] for r in test])
    scores=np.column_stack((np.ones(len(test)),(sample-mean)/sd))@beta
    for r,v in zip(test,scores):r['ridge']=float(v)
    threshold=sorted(train,key=lambda r:r['y'])[int(.75*len(train))]['y']
    assert abs(threshold+1.6)<1e-9
    summary={}
    for method in ('ridge','p'):
        chosen=sorted(test,key=lambda r:r[method],reverse=True)[:20]
        truth=np.array([r['y'] for r in test])
        predict=np.array([r[method] for r in test])
        summary[method]={'top20_hits':int(sum(r['y']>=threshold for r in chosen)),
                         'top20_measured_mean':float(np.mean([r['y'] for r in chosen])),
                         'test_mae':float(np.mean(np.abs(truth-predict))),
                         'test_rmse':float(np.sqrt(np.mean((truth-predict)**2)))}
    with (args.output/'test_predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['name','smiles','measured_logS','ridge','published_ESOL','hit'])
        w.writerows([r['name'],r['smiles'],r['y'],r['ridge'],r['p'],int(r['y']>=threshold)] for r in test)
    report={'n_source':len(source),'n_train':len(train),'n_test':len(test),
            'threshold':threshold,'coefficients':beta.tolist(),'strategies':summary,
            'test_random_hit_fraction':float(np.mean([r['y']>=threshold for r in test])),
            'note':'Retrospective real assays of known compounds; no prospective validation'}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
