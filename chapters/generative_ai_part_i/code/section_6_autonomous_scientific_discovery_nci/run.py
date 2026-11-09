"""Real NCI structure molecular candidate generation with learned surrogate versus random-search control.
Scientific scope: RDKit calculated physicochemical proxy, NOT biological discovery or wet-lab activity.
"""
import csv, json, hashlib, random, argparse
from pathlib import Path
import numpy as np
from rdkit import Chem, RDConfig, RDLogger, DataStructs
from rdkit.Chem import QED, Descriptors, rdFingerprintGenerator, Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
RDLogger.DisableLog('rdApp.error')
SEEDS=(11,23,37)
GEN=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=512)

def fp(mol):return np.asarray(GEN.GetFingerprint(mol),dtype='float32')
def objective(mol):
    mw=Descriptors.MolWt(mol); logp=Descriptors.MolLogP(mol)
    return float(QED.qed(mol)-.003*max(mw-450,0)-.07*max(logp-4,0))
def canonical(m):return Chem.MolToSmiles(m,canonical=True,isomericSmiles=False)
def mutate(smiles,rng):
    parent=Chem.MolFromSmiles(smiles)
    if parent is None:return None
    atoms=parent.GetNumAtoms()
    if atoms>65:return None
    edit=Chem.RWMol(parent)
    operation=rng.choice(['replace','attach','delete'])
    try:
        if operation=='replace':
            available=[a.GetIdx() for a in edit.GetAtoms() if a.GetAtomicNum() in (6,7,8,9,16) and not a.GetIsAromatic()]
            if not available:return None
            i=rng.choice(available);edit.GetAtomWithIdx(i).SetAtomicNum(rng.choice([6,7,8,9]))
        elif operation=='attach':
            available=[a.GetIdx() for a in edit.GetAtoms() if not a.GetIsAromatic() and a.GetAtomicNum() in (6,7,8)]
            if not available:return None
            i=rng.choice(available);j=edit.AddAtom(Chem.Atom(rng.choice([6,6,6,7,8])))
            edit.AddBond(i,j,Chem.BondType.SINGLE)
        else:
            terminal=[a.GetIdx() for a in edit.GetAtoms() if a.GetDegree()==1 and not a.IsInRing() and not a.GetIsAromatic()]
            if not terminal:return None
            edit.RemoveAtom(rng.choice(terminal))
        m=edit.GetMol();Chem.SanitizeMol(m)
        return canonical(m)
    except Exception:return None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--budget',type=int,default=160);ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'results');a=ap.parse_args()
    out=a.output;figdir=out/'figures';figdir.mkdir(parents=True,exist_ok=True)
    path=Path(RDConfig.RDDataDir)/'NCI'/'first_5K.smi';sh=hashlib.sha256(path.read_bytes()).hexdigest()
    allsmiles=[];seen=set()
    for line in path.read_text().splitlines():
        parts=line.split()
        if not parts:continue
        mol=Chem.MolFromSmiles(parts[0]);
        if mol is None or mol.GetNumAtoms()>65:continue
        s=canonical(mol)
        if s in seen:continue
        seen.add(s);allsmiles.append(s)
    assert len(allsmiles)>3000
    trainidx,testidx=train_test_split(np.arange(len(allsmiles)),test_size=.2,random_state=20261109)
    mols=[Chem.MolFromSmiles(s) for s in allsmiles]
    x=np.stack([fp(m) for m in mols]);y=np.array([objective(m) for m in mols])
    surrogate=RandomForestRegressor(n_estimators=35,max_depth=12,min_samples_leaf=2,n_jobs=2,random_state=20261109)
    surrogate.fit(x[trainidx],y[trainidx]);yhat=surrogate.predict(x[testidx]);
    held={'n_train':len(trainidx),'n_test':len(testidx),'MAE':float(mean_absolute_error(y[testidx],yhat)),'R2':float(r2_score(y[testidx],yhat))}
    with (out/'real_nci_test_predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['source_index','canonical_smiles','actual_descriptor_objective','surrogate_prediction']);w.writerows((int(i),allsmiles[i],float(y[i]),float(p)) for i,p in zip(testidx,yhat))
    train_set={allsmiles[i] for i in trainidx};whole_set=set(allsmiles)
    pool=[allsmiles[i] for i in trainidx];poolscore={s:float(y[i]) for i,s in zip(trainidx,pool)}
    bestsource=max(pool,key=lambda s:poolscore[s]);records=[];summaries=[];bestgenerated=[]
    for seed in SEEDS:
        for arm in ('surrogate_guided','random_control'):
            rng=random.Random(seed+1000*(arm=='random_control'))
            cache=set(pool);candidates=[];scores=[];predscores={};n_attempts=0
            while len(candidates)<a.budget and n_attempts<a.budget*100:
                n_attempts+=1
                parents=pool if len(candidates)<24 or arm=='random_control' else pool+[s for s in candidates[-32:]]
                if arm=='surrogate_guided' and len(candidates)>24:
                    parent=rng.choice(sorted(candidates,key=lambda s:predscores[s],reverse=True)[:max(12,len(candidates)//4)])
                else:parent=rng.choice(parents)
                proposal=[]
                for _ in range(8 if arm=='surrogate_guided' else 1):
                    s=mutate(parent,rng)
                    if s and s not in cache and s not in proposal and s not in whole_set:proposal.append(s)
                if not proposal:continue
                if arm=='surrogate_guided':
                    f=np.stack([fp(Chem.MolFromSmiles(s)) for s in proposal]);s=proposal[int(np.argmax(surrogate.predict(f)))]
                else:s=proposal[0]
                mol=Chem.MolFromSmiles(s);score=objective(mol)
                cache.add(s);candidates.append(s);scores.append(score);predscores[s]=float(surrogate.predict(fp(mol)[None])[0])
                records.append({'seed':seed,'arm':arm,'step':len(candidates),'smiles':s,'surrogate_score':predscores[s],'true_descriptor_objective':score,'QED':float(QED.qed(mol)),'MW':float(Descriptors.MolWt(mol)),'LogP':float(Descriptors.MolLogP(mol)),'novel_to_full_NCI':int(s not in whole_set),'best_so_far':max(scores)})
            assert len(candidates)==a.budget,(seed,arm,len(candidates))
            bestidx=int(np.argmax(scores));bestgenerated.append((seed,arm,candidates[bestidx],scores[bestidx]))
            scaffolds={MurckoScaffold.MurckoScaffoldSmilesFromSmiles(s) for s in candidates}
            summaries.append({'seed':seed,'arm':arm,'candidate_budget':len(candidates),'best_true_descriptor_score':float(np.max(scores)),'mean_true_descriptor_score':float(np.mean(scores)),'median_true_descriptor_score':float(np.median(scores)),'n_distinct_scaffolds':len(scaffolds),'n_novel_to_full_NCI':sum(s not in whole_set for s in candidates),'n_valid_sanitized':len(candidates)})
            print('COMPLETE',seed,arm,summaries[-1],flush=True)
    with (out/'generated_candidates.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    with (out/'top_molecules.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['seed','arm','smiles','descriptor_objective']);w.writerows(bestgenerated)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for arm in ('surrogate_guided','random_control'):
        matrix=np.stack([[r['best_so_far'] for r in records if r['arm']==arm and r['seed']==seed] for seed in SEEDS])
        label='Learned-surrogate guided' if arm=='surrogate_guided' else 'Random molecular edits'
        ax.plot(np.arange(1,a.budget+1),matrix.mean(0),label=label)
        ax.fill_between(np.arange(1,a.budget+1),matrix.min(0),matrix.max(0),alpha=.15)
    ax.set(xlabel='Sanitized generated candidates evaluated',ylabel='Best generated penalized QED (RDKit calculation)');ax.legend(frameon=False);fig.tight_layout();fig.savefig(figdir/'molecule_discovery_search_curves.png',dpi=190);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.0),constrained_layout=True)
    for j,k in enumerate(['mean_true_descriptor_score','n_distinct_scaffolds']):
        for i,arm in enumerate(('surrogate_guided','random_control')):
            vals=np.array([row[k] for row in summaries if row['arm']==arm]);axes[j].scatter(i+np.array([-.08,0,.08]),vals,s=40);axes[j].hlines(vals.mean(),i-.15,i+.15,lw=2)
        axes[j].set_xticks([0,1],['Surrogate guided','Random edits']);axes[j].set_ylabel(k.replace('_',' '));axes[j].grid(axis='y',alpha=.15)
    fig.savefig(figdir/'molecule_discovery_quality_diversity.png',dpi=190);plt.close(fig)
    ordered=sorted([r for r in records if r['arm']=='surrogate_guided'],key=lambda r:r['true_descriptor_objective'],reverse=True)
    picks=[];chem=[]
    for r in ordered:
        if r['smiles'] in chem:continue
        chem.append(r['smiles']);picks.append(Chem.MolFromSmiles(r['smiles']))
        if len(picks)==8:break
    gallery=Draw.MolsToGridImage(picks,molsPerRow=4,subImgSize=(250,180),legends=[f'QED*={objective(m):.3f}' for m in picks]);gallery.save(figdir/'generated_nci_candidate_structures.png')
    manifest={'dataset':'real RDKit bundled NCI first_5K.smi','source_sha256':sh,'n_source_molecules_unique_filtered':len(allsmiles),'n_train':len(trainidx),'n_heldout_test':len(testidx),'train_test_split_seed':20261109,'surrogate_metrics_on_real_NCI':held,'generative_seeds':list(SEEDS),'valid_new_candidate_budget_per_arm_seed':a.budget,'assay':'NONE - optimized QED/MW/LogP are deterministic RDKit descriptors, not experimental measurements','candidate_novelty':'different canonical SMILES than full source collection; no claim of global novelty','limitations':['No external assays or biological target','Generator uses hand-designed graph edits, not neural decoder','Same property used both surrogate training and objective; synthetic design may be overoptimized','One NCI molecular source corpus','No synthesizeability testing','Only 3 seeds and fixed candidate budget']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'real_source_test':held,'summaries':summaries},indent=2))
if __name__=='__main__':main()
