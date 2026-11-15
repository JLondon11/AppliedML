"""Validate exact Chapter 6 certification CSV shape before scientific claims.

Unlike DictReader alone, this rejects any extra PASS that shifts the science
and production statuses. It validates registry integrity, not model execution.
"""
import argparse,csv,json
from pathlib import Path
FIELDS=['unit_id','parent_study','level','certified_scope',*[f'R{i}' for i in range(1,11)],
        'scientific_status','production_status','evidence_limits']
def validate(path):
    with Path(path).open(newline='',encoding='utf-8') as stream:
        raw=list(csv.reader(stream,strict=True))
    if not raw or raw[0]!=FIELDS:
        raise ValueError('Chapter 6 17-column scientific/production schema does not match')
    if len(raw)<13:
        raise ValueError('Expected 12 or more recorded scientific scope units')
    observations=[]
    for line,record in enumerate(raw[1:],2):
        if len(record)!=len(FIELDS):
            raise ValueError(f'CSV line {line}: exactly 17 fields required, got {len(record)}')
        observations.append(dict(zip(FIELDS,record)))
    ids=[r['unit_id'] for r in observations]
    if ids!=[f'CH6-D{i:02d}' for i in range(1,len(observations)+1)]:
        raise ValueError('Original scoped scientific ID continuity/order violated')
    result={'scientifically_certified':0,'production_hold':0,'level_A':0,'level_B':0}
    for r in observations:
        ident=r['unit_id']
        expected_level='B' if ident=='CH6-D04' else 'A'
        if r['level']!=expected_level:
            raise ValueError(f'{ident}: level out of documented proportional scope')
        if not r['parent_study'].strip() or not r['certified_scope'].strip() or not r['evidence_limits'].strip():
            raise ValueError(f'{ident}: scientific title/scope/limitations incomplete')
        if any(r[f'R{i}']!='PASS' for i in range(1,11)):
            raise ValueError(f'{ident}: scientific full-certification requirement not passed')
        if r['scientific_status']!='SCIENTIFIC_CERTIFIED':
            raise ValueError(f'{ident}: statuses malformed or science not certified')
        if r['production_status']!='PRODUCTION_HOLD':
            raise ValueError(f'{ident}: unsupported production approval; current manuscript missing')
        result['scientifically_certified']+=1
        result['production_hold']+=1
        result['level_'+r['level']]+=1
    return {'schema_fields':len(FIELDS),'units':len(observations),
            **result,'parent_application_certified_by_this_register':0,
            'production_release':'HOLD',
            'note':'This validates record integrity, not source models or PDF production'}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--register',required=True,type=Path)
    a=parser.parse_args()
    print(json.dumps(validate(a.register),indent=2))
