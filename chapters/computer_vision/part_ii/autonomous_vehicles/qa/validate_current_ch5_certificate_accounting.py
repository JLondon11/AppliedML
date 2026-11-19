"""Cross-register Chapter 5 proportional science certification gate.

Checks *both* the eleven full named parent applications and three embedded
Level A experiments, keeping original science distinct from production approval.
No inferred certification; does not independently rerun original models.
"""
import argparse,csv,json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PARENTS=ROOT/'CH5_SCIENTIFIC_VS_PRODUCTION_CERTIFICATION_20261112.csv'
DEMOS=ROOT/'CH5_LEVEL_A_EMBEDDED_DEMONSTRATION_CERTIFICATIONS_20261112.csv'

def read_exact(path,fields):
    with Path(path).open(newline='',encoding='utf-8') as f:
        reader=csv.reader(f,strict=True)
        raw=list(reader)
    if not raw or raw[0]!=fields:raise ValueError(str(path)+': unexpected exact CSV schema')
    if any(len(row)!=len(fields) for row in raw[1:]):
        raise ValueError(str(path)+': malformed or shifted certification row')
    return [dict(zip(fields,row)) for row in raw[1:]]

def check(parents=PARENTS,demos=DEMOS):
    r=[f'R{i}' for i in range(1,11)]
    p_fields=['unit_id','level','unit',*r,'primary_justification','scientific_status','production_status']
    d_fields=['demonstration_id','parent_unit','level','certified_scientific_scope',*r,
              'scientific_status','production_status','limits']
    p=read_exact(parents,p_fields);d=read_exact(demos,d_fields)
    if len(p)!=11 or [x['unit_id'] for x in p]!=[f'CV5-CS{i:02d}' for i in range(1,12)]:
        raise ValueError('Expected eleven distinct parent case studies')
    if len(d)!=3 or [x['demonstration_id'] for x in d]!=[f'CV5-D{i:02d}' for i in range(1,4)]:
        raise ValueError('Expected exactly three original Level A embedded studies')
    expected_parent={'CV5-D01':'CV5-CS03','CV5-D02':'CV5-CS11','CV5-D03':'CV5-CS06'}
    parent_certified=[]
    for x in p:
        ident=x['unit_id'];expected_level='C' if ident in ('CV5-CS04','CV5-CS05') else 'B'
        if x['level']!=expected_level:raise ValueError(ident+' incorrect parent level')
        if x['production_status']!='PRODUCTION_HOLD':raise ValueError(ident+' unsupported manuscript production approval')
        if x['scientific_status'] not in ('SCIENTIFIC_CERTIFIED','SCIENTIFIC_HOLD'):
            raise ValueError(ident+' invalid scientific status')
        n=sum(x[k]=='PASS' for k in r)
        if (n==10)!=(x['scientific_status']=='SCIENTIFIC_CERTIFIED'):
            raise ValueError(ident+' certificate inconsistent with original ten requirement statuses')
        if not x['primary_justification'].strip():raise ValueError(ident+' missing scientific provenance')
        if x['scientific_status']=='SCIENTIFIC_CERTIFIED':parent_certified.append(ident)
    if parent_certified!=['CV5-CS09']:
        raise ValueError('Full parent certificates require new independent genuine experiments and explicit registry updates')
    for x in d:
        ident=x['demonstration_id']
        if x['parent_unit']!=expected_parent[ident] or x['level']!='A':
            raise ValueError(ident+' linked to incorrect parent scientific scope')
        if any(x[k]!='PASS' for k in r):
            raise ValueError(ident+' missing original-source requirement evidence')
        if (x['scientific_status'],x['production_status'])!=('SCIENTIFIC_CERTIFIED','PRODUCTION_HOLD'):
            raise ValueError(ident+' has unsupported publication gate')
        if len(x['limits'].strip())<90 or len(x['certified_scientific_scope'])<40:
            raise ValueError(ident+' insufficient limited experimental claims')
        if next(z for z in p if z['unit_id']==x['parent_unit'])['scientific_status']!='SCIENTIFIC_HOLD':
            raise ValueError(ident+' parent status changed without separate full case study evidence')
    yolo=next(x for x in d if x['demonstration_id']=='CV5-D01')
    if '23 unique' not in yolo['limits'] or '24' not in yolo['certified_scientific_scope']:
        raise ValueError('Real VisDrone source duplicate is not disclosed')
    stereo=next(x for x in d if x['demonstration_id']=='CV5-D03')
    if 'NO camera-LiDAR fusion' not in stereo['limits']:
        raise ValueError('Classical original stereo cannot certify fused camera-LiDAR model')
    return {'parent_applications':11,'certified_parent_applications':1,
            'parent_ids_certified':parent_certified,
            'separate_level_a_embedded_certificates':3,
            'scientifically_certified_distinct_scopes_total':4,
            'parent_level_B_count':9,'parent_level_C_count':2,
            'experimental_units_not_silently_promoted':True,
            'all_production_approvals':0,
            'real_data_duplicate_yolo_file_count':24,
            'real_data_unique_yolo_image_contents':23,
            'chapter_full_scientific_approval':False,
            'chapter_full_manuscript_approval':False,
            'verification_limits':'Record structure and scope reconciliation; archived original neural model data not recomputed by this script.'}
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--parents',type=Path,default=PARENTS)
    p.add_argument('--demos',type=Path,default=DEMOS)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    report=check(a.parents,a.demos)
    body=json.dumps(report,indent=2)+'\n'
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(body)
    print(body,end='')
