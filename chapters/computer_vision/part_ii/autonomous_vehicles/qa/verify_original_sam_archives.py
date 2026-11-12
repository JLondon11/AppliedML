"""Validate three original SAM/Mask R-CNN GitHub Actions ZIP archives.

Requires actual original workflow ZIPs. Reconciles per-object observations,
source-image identity and original masks/figure inventory; not a model rerun.
"""
import argparse,csv,hashlib,io,json,math,statistics,zipfile
from pathlib import Path

def load(path):
    path=Path(path)
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate ZIP entries')
        for f in ('per_object_metrics.csv','metrics_summary.json','selection_provenance.json'):
            if f not in names:raise ValueError(f'Missing measured source file {f}')
        rows=list(csv.DictReader(io.StringIO(z.read('per_object_metrics.csv').decode())))
        summary=json.loads(z.read('metrics_summary.json'))
        provenance=json.loads(z.read('selection_provenance.json'))
        selected=provenance['image_ids']
        if len(selected)!=len(set(selected)) or set(selected)!={int(r['image_id']) for r in rows}:
            raise ValueError('Recorded original image IDs do not reconcile')
        hashed=provenance.get('original_image_sha256',{})
        if set(hashed)!={f'{i:012d}.jpg' for i in selected}:
            raise ValueError('Original JPEG identity/hash inventory mismatch')
        if not all(len(h)==64 for h in hashed.values()):raise ValueError('Bad original image hash')
        if len(rows)!=len(set((r['image_id'],r['annotation_id']) for r in rows)):
            raise ValueError('Duplicate observed original annotation IDs')
        figs=[x for x in names if x.startswith('Figures/') and x.endswith('.png')]
        if not figs:raise ValueError('Missing original model output panels')
    return {'rows':rows,'summary':summary,'provenance':provenance,
            'zip_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'archive_size_bytes':path.stat().st_size,'file_count':len(names),
            'png_figure_count':len(figs)}

def assert_equal(x,y,description):
    if not math.isclose(float(x),float(y),rel_tol=0,abs_tol=2e-12):
        raise ValueError(f'{description} does not reconcile: {x} vs {y}')

def verify(s18,s24,mask):
    for s in (s18,s24):
        rows=s['rows'];meta=s['summary']
        if len(rows)!=meta['instances'] or len(s['provenance']['image_ids'])!=meta['images']:
            raise ValueError('SAM original source sample count mismatch')
        assert_equal(statistics.mean(float(r['sam_iou']) for r in rows),meta['sam_mean_iou'],'SAM IoU')
        assert_equal(statistics.mean(float(r['sam_dice']) for r in rows),meta['sam_mean_dice'],'SAM Dice')
        pairs=[r for r in rows if r['grabcut_valid']=='True']
        if len(pairs)!=meta['grabcut_paired_samples']:raise ValueError('Invalid paired count')
        assert_equal(statistics.mean(float(r['sam_iou']) for r in pairs),meta['paired_sam_iou'],'paired SAM IoU')
        assert_equal(statistics.mean(float(r['grabcut_iou']) for r in pairs),meta['paired_grabcut_iou'],'paired GrabCut IoU')
    if set(s18['provenance']['image_ids']) & set(s24['provenance']['image_ids']):
        raise ValueError('Overlapping 18/24 original images')
    if s18['provenance']['original_image_sha256']!=mask['provenance']['original_image_sha256']:
        raise ValueError('SAM and Mask R-CNN same-image provenance mismatch')
    pairs=lambda s:{(r['image_id'],r['annotation_id']) for r in s['rows']}
    if pairs(s18)!=pairs(mask):raise ValueError('SAM and Mask R-CNN annotation IDs differ')
    mr=mask['rows'];ms=mask['summary']
    if len(mr)!=ms['objects'] or len(mask['provenance']['image_ids'])!=ms['images']:
        raise ValueError('Mask R-CNN sample count mismatch')
    matched=sum(r['matched']=='True' for r in mr)
    if matched!=ms['matched']:raise ValueError('Mask R-CNN TP count mismatch')
    mask_iou=statistics.mean(float(r['mask_iou']) for r in mr)
    assert_equal(mask_iou,ms['mean_object_iou'],'Mask R-CNN mean IoU')
    rows=s18['rows']+s24['rows'];paired=[r for r in rows if r['grabcut_valid']=='True']
    results={
        'original_coco_images':len(s18['provenance']['image_ids'])+len(s24['provenance']['image_ids']),
        'sam_original_objects':len(rows),
        'paired_objects':len(paired),
        'sam_mean_iou':statistics.mean(float(r['sam_iou']) for r in rows),
        'sam_mean_dice':statistics.mean(float(r['sam_dice']) for r in rows),
        'paired_sam_iou':statistics.mean(float(r['sam_iou']) for r in paired),
        'paired_grabcut_iou':statistics.mean(float(r['grabcut_iou']) for r in paired),
        'paired_mean_iou_difference':statistics.mean(float(r['sam_iou'])-float(r['grabcut_iou']) for r in paired),
        'maskrcnn_original_objects':len(mr),'maskrcnn_matched_objects':matched,'maskrcnn_mean_iou':mask_iou}
    expected={'original_coco_images':42,'sam_original_objects':147,'paired_objects':99,
      'sam_mean_iou':0.7786187121390236,'sam_mean_dice':0.8577038635232835,
      'paired_sam_iou':0.8062725695842572,'paired_grabcut_iou':0.4864770976716845,
      'paired_mean_iou_difference':0.3197954719125727,
      'maskrcnn_original_objects':59,'maskrcnn_matched_objects':56,'maskrcnn_mean_iou':0.7453008745762045}
    for k,expected_value in expected.items():assert_equal(results[k],expected_value,k)
    return results

def main():
    p=argparse.ArgumentParser()
    for name in ('sam-18','sam-24','maskrcnn-18'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    artifacts={name:load(getattr(a,name)) for name in ('sam_18','sam_24','maskrcnn_18')}
    results=verify(artifacts['sam_18'],artifacts['sam_24'],artifacts['maskrcnn_18'])
    report={'status':'PASS_ORIGINAL_PER_OBJECT_RECONCILIATION',
       'artifacts':{k:{x:v[x] for x in ('zip_sha256','archive_size_bytes','file_count','png_figure_count')} for k,v in artifacts.items()},
       'measured':results,
       'boundaries':'Does not independently rerun model inference, inspect full manuscript PDF or validate all mask panel pixels. Do not certify R9 without final figure-to-manuscript review.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
