"""Validate independently scoped Chapter 6 Level A certificates and original evidence.

Checks register integrity and independently saved Brownlee 21-fold measured records;
does not falsely certify broader parent case studies or final manuscript production.
"""
import csv,math,json
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parent
expected={f'CH6-D{i:02d}' for i in range(1,13)}
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

# CH6-D10: source-pinned retrospective ESOL on 226 actual assay labels.
E=ROOT.parent/'code'/'section_6_autonomous_scientific_discovery_nci'/'retrospective_esol'
with (E/'verified_original_ESOL_226_test_predictions.csv').open(newline='',encoding='utf-8') as f:
    esol=list(csv.DictReader(f))
if len(esol)!=226 or len({r['smiles'] for r in esol})!=226:
    raise ValueError('Expected exactly 226 unique original ESOL held-out compounds')
for key,target,hits in [('ridge',1.1795769332543176,14),
                         ('published_ESOL',.8755685030537721,17)]:
    val=math.sqrt(sum((float(r[key])-float(r['measured_logS']))**2 for r in esol)/226)
    if abs(val-target)>1e-9:raise ValueError('Original ESOL RMSE mismatch '+key)
    top=sorted(esol,key=lambda r:float(r[key]),reverse=True)[:20]
    if sum(float(r['measured_logS'])>=-1.6 for r in top)!=hits:
        raise ValueError('Original ESOL top-20 measured hit mismatch '+key)
if not all(int(r['hit'])==int(float(r['measured_logS'])>=-1.6) for r in esol):
    raise ValueError('ESOL original assay threshold labels mismatch')

# CH6-D04: recompute authentic matched-seed NEU negative result from the
# ORIGINAL 12 observed model-evaluation rows; do not treat test photos as
# independent across repeated seeds or make a cross-factory claim.
import importlib.util
_NEUDIR = ROOT.parent / 'code' / 'section_8_8_2_neu_defect_benchmark'
_spec = importlib.util.spec_from_file_location('ch6_neu_seed_audit', _NEUDIR/'audit_paired_seed_uncertainty.py')
if _spec is None or _spec.loader is None:
    raise ValueError('Cannot load original NEU seed evidence auditor')
_neu_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_neu_module)
_neu_report, _neu_pairs = _neu_module.analyze(_NEUDIR/'replicated_seed_metrics_pass13.csv')
if len(_neu_pairs) != 12 or _neu_report['source_rows'] != 12:
    raise ValueError('NEU measured seed pairing incomplete')
for metric, comparator, target in (
    ('test_accuracy', 'real_only', -0.19259259263333336),
    ('test_macro_f1', 'real_only', -0.2061735354333333),
    ('test_accuracy', 'real_repeat_augmented', -0.2694444444666667),
    ('test_macro_f1', 'real_repeat_augmented', -0.2870857760666667),
):
    observed = _neu_report['metrics'][metric][comparator]
    if abs(observed['mean_difference'] - target) > 1e-10:
        raise ValueError('Original NEU paired scientific effect changed: '+metric+'/'+comparator)
    if observed['negative_difference_seeds'] != 3 or observed['one_sided_exact_sign_test_tail'] != .125:
        raise ValueError('NEU small-sample seed effect overstated')

# CH6-D12: independently reconcile original per-class digit generator classifier outcomes.
DIGITS=ROOT.parent/'code'/'section_6_20_1_latent_diffusion_digits'/'figure_6_44_class_counts_pass27.csv'
with DIGITS.open(newline='') as fh:
    digit_rows=list(csv.DictReader(fh))
if len(digit_rows)!=22 or {(r['arm'],r['label']) for r in digit_rows}!={(arm,label) for arm in ('conditional','unconditional') for label in ('overall',*map(str,range(10)))}:
    raise ValueError('Missing original 22 digit-generator classifier summary records')
for arm,expected_hits in (('conditional',778),('unconditional',94)):
    classes=[r for r in digit_rows if r['arm']==arm and r['label']!='overall']
    overall=[r for r in digit_rows if r['arm']==arm and r['label']=='overall']
    if len(classes)!=10 or len(overall)!=1:raise ValueError('Missing digit labels')
    if any(int(r['n'])!=100 for r in classes):raise ValueError('Incorrect original per-digit count')
    if sum(int(r['correct']) for r in classes)!=expected_hits:
        raise ValueError('Original diffusion predicted-class results do not reconcile')
    if int(overall[0]['correct'])!=expected_hits or int(overall[0]['n'])!=1000:
        raise ValueError('Original diffusion overall total differs from per-digit counts')

# CH6-D12: reconcile historical archived model and separately executed current
# original-data CI model rather than enforcing false binary/numerical identity.
_DIFF = ROOT.parent/'code'/'section_6_20_1_latent_diffusion_digits'
_diff_spec = importlib.util.spec_from_file_location(
    'ch6_digit_run_diff', _DIFF/'verify_independent_digit_run_difference.py')
if _diff_spec is None or _diff_spec.loader is None:
    raise ValueError('Cannot load genuine digit-run reconciliation verifier')
_diff_module = importlib.util.module_from_spec(_diff_spec)
_diff_spec.loader.exec_module(_diff_module)
_delta = _diff_module.reconcile(
    _DIFF/'figure_6_44_class_counts_pass27.csv',
    _DIFF/'qa'/'ci_44209629586'/'original_executed_class_counts.csv')
if _delta['archive_conditional_correct']!=778 or _delta['current_conditional_correct']!=779:
    raise ValueError('Historic and new authentic digit generator results were conflated')

print(json.dumps({'scoped_scientific_certificates':12,'levels':{'A':11,'B':1},'original_plant_heldouts':21,
                  'brownlee_90pct_coverage':coverage,'chapter6_parent_applications_certified_by_this_register':0,
                  'production_status':'HOLD'},indent=2))
