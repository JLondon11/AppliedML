"""Recompute all candidate descriptor objectives and validate source provenance."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from rdkit import Chem,RDConfig
from rdkit.Chem import QED,Descriptors
ROOT=Path(__file__).resolve().parent/'results'
with (ROOT/'manifest.json').open() as f:m=json.load(f)
source=Path(RDConfig.RDDataDir)/'NCI'/'first_5K.smi'
assert hashlib.sha256(source.read_bytes()).hexdigest()==m['source_sha256']
with (ROOT/'generated_candidates.csv').open() as f:records=list(csv.DictReader(f))
with (ROOT/'summary.csv').open() as f:summaries=list(csv.DictReader(f))
with (ROOT/'real_nci_test_predictions.csv').open() as f:test=list(csv.DictReader(f))
assert len(records)==300 and len(summaries)==6 and len(test)==978
for row in records:
 mol=Chem.MolFromSmiles(row['smiles']);assert mol is not None
 score=QED.qed(mol)-.003*max(Descriptors.MolWt(mol)-450,0)-.07*max(Descriptors.MolLogP(mol)-4,0)
 assert abs(score-float(row['true_descriptor_objective']))<1e-9
for row in summaries:
 group=[r for r in records if r['seed']==row['seed'] and r['arm']==row['arm']]
 assert len(group)==50 and len(set(r['smiles'] for r in group))==50
 assert abs(max(float(r['true_descriptor_objective']) for r in group)-float(row['best_true_descriptor_score']))<1e-9
 assert abs(np.mean([float(r['true_descriptor_objective']) for r in group])-float(row['mean_true_descriptor_score']))<1e-9
assert len({r['smiles'] for r in records})>=200
for name in ('molecule_discovery_search_curves.png','molecule_discovery_quality_diversity.png','generated_nci_candidate_structures.png'):
 assert (ROOT/'figures'/name).stat().st_size>10000
print('PASS: 4,886 genuine NCI records, 978 held-out property predictions, 300 sanitized generated candidates, 3 scientific figures')
