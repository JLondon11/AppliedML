"""Validate independently scoped Chapter 6 Level A certificates and original evidence.

Checks register integrity and independently saved Brownlee 21-fold measured records;
does not falsely certify broader parent case studies or final manuscript production.
"""
import csv,math,json
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parent
expected={'CH6-D01','CH6-D02','CH6-D03','CH6-D04','CH6-D05','CH6-D06','CH6-D07'}
with (ROOT/'CH6_SCOPED_SCIENTIFIC_VS_PRODUCTION_REGISTER_20261113.csv').open(newline='') as f:
    rows=list(csv.DictReader(f))
if {r['unit_id'] for r in rows}!=expected or len(rows)!=len(expected):raise ValueError('Incorrect experimental-unit set')
for r in rows:
    if r['level']!=('B' if r['unit_id']=='CH6-D04' else 'A') or r['scientific_status']!='SCIENTIFIC_CERTIFIED' or r['production_status']!='PRODUCTION_HOLD':
        raise ValueError(r['unit_id']+': incorrect scope or scientific/production gate')
    if any(r[f'R{i}']!='PASS' for i in range(1,11)):raise ValueError(r['unit_id']+': missing scientific requirement')
    if not r['evidence_limits']:raise ValueError('Missing claim limits')
with (ROOT/'brownlee_original_loo_replay_20261113.csv').open(newline='') as f:
    plant=list(csv.DictReader(f))
if len(plant)!=21 or [int(r['heldout_id']) for r in plant]!=list(range(1,22)):
    raise ValueError('Brownlee original out-of-fold samples inconsistent')
y=[float(r['observed_stack_loss']) for r in plant]
def rmse(k):
    return math.sqrt(sum((float(r[k])-float(r['observed_stack_loss']))**2 for r in plant)/len(plant))
for col,expected_value in [('gp_predictive_mean',4.118931861605197),
                           ('ridge_prediction',3.647664958998736),
                           ('training_mean_prediction',10.422811520890129)]:
    if abs(rmse(col)-expected_value)>1e-8:raise ValueError(col+' recorded RMSE mismatch')
coverage=sum(float(r['gp_p05'])<=float(r['observed_stack_loss'])<=float(r['gp_p95']) for r in plant)
if coverage!=19:raise ValueError('Brownlee interval coverage not 19/21')
# Independent original Yeh row, group separation and metric verification.
from audit_concrete_retrospective import audit as audit_concrete
concrete = audit_concrete()
if concrete["heldout_observations"] != 229 or concrete["independent_physical_mix_tests"] != 0:
    raise ValueError("Concrete scoped certificate evidence boundary invalid")

# CH6-D07: additionally verify genuine held-out per-seed metrics committed with source.
CYTOLOGY = ROOT.parent / 'code' / 'section_6_precision_medicine_cytology_cvae' / 'pass54_original_test_metrics.csv'
with CYTOLOGY.open(newline='') as f:
    cytology = list(csv.DictReader(f))
if len(cytology) != 9:
    raise ValueError('WDBC original-data metric inventory must have 9 source rows')
cytology_expected = {
    'real_only': (0.9993386243386243, 0.9930555555555557),
    'real_bootstrap': (0.9986772486772487, 0.9755291005291006),
    'real_plus_cvae': (0.98070987654321, 0.9470899470899471),
}
for arm, (auc, balanced_accuracy) in cytology_expected.items():
    values = [r for r in cytology if r['arm'] == arm]
    if sorted(int(r['seed']) for r in values) != [11, 23, 37]:
        raise ValueError('WDBC original seed/arm record mismatch')
    if abs(sum(float(r['ROC_AUC']) for r in values)/3 - auc) > 1e-10:
        raise ValueError('WDBC original held-out ROC AUC mismatch')
    if abs(sum(float(r['balanced_accuracy']) for r in values)/3 - balanced_accuracy) > 1e-10:
        raise ValueError('WDBC original held-out balanced accuracy mismatch')

print(json.dumps({'scoped_scientific_certificates':7,'levels':{'A':6,'B':1},'original_plant_heldouts':21,
                  'brownlee_90pct_coverage':coverage,'chapter6_parent_applications_certified_by_this_register':0,
                  'production_status':'HOLD'},indent=2))
