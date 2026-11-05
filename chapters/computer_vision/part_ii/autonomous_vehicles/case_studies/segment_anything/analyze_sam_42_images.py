#!/usr/bin/env python3
"""Pooled descriptive real-source SAM benchmark analysis.

Inputs: ORIGINAL successful Github Actions result artifacts, not synthetic metrics.
The 18-image run is seeded, 24-image run is sorted by original COCO ID.
Their image IDs must not overlap; paired comparator is GrabCut under shared GT box.
Image-cluster resampling quantifies conditional variability, NOT official COCO AP.
"""
import argparse,csv,io,json,zipfile,hashlib
from pathlib import Path
import numpy as np

def rows_from_zip(path):
    with zipfile.ZipFile(path) as z:
        rows=list(csv.DictReader(io.StringIO(z.read('per_object_metrics.csv').decode())))
        provenance=json.loads(z.read('selection_provenance.json'))
        summary=json.loads(z.read('metrics_summary.json'))
    for r in rows:
        r['image_id']=int(r['image_id'])
        r['annotation_id']=int(r['annotation_id'])
        for k in ('sam_iou','sam_dice','grabcut_iou','grabcut_dice'):
            r[k]=float(r[k])
        r['grabcut_valid']=r['grabcut_valid'].lower()=='true'
    return rows,provenance,summary

def analyze(paths,out,seed=20261105,draws=10000):
    two=[rows_from_zip(p) for p in paths]
    ids=[set(x[1]['image_ids']) for x in two]
    if ids[0]&ids[1]:raise ValueError('Overlapping image identifiers: cannot claim independent scenes')
    if any(x[2]['checkpoint_sha256']!=two[0][2]['checkpoint_sha256'] for x in two):
        raise ValueError('Different SAM checkpoint across two experiments')
    rows=[]
    for name,(samples,provenance,summary) in zip(('seeded_18','sorted_24'),two):
        for r in samples:rows.append(dict(r,experiment=name))
    if len({(r['image_id'],r['annotation_id']) for r in rows})!=len(rows):
        raise ValueError('Repeated instance annotation identifier')
    images=sorted(set(r['image_id'] for r in rows));groups={i:[r for r in rows if r['image_id']==i] for i in images}
    paired=[r for r in rows if r['grabcut_valid']]
    def stats(rs):
        valid=[r for r in rs if r['grabcut_valid']]
        return {'sam_mean_iou':np.mean([r['sam_iou'] for r in rs]),
                'sam_mean_dice':np.mean([r['sam_dice'] for r in rs]),
                'paired_sam_iou':np.mean([r['sam_iou'] for r in valid]) if valid else np.nan,
                'paired_grabcut_iou':np.mean([r['grabcut_iou'] for r in valid]) if valid else np.nan,
                'paired_difference':np.mean([r['sam_iou']-r['grabcut_iou'] for r in valid]) if valid else np.nan}
    observed=stats(rows)
    rng=np.random.default_rng(seed);boot={k:[] for k in observed}
    for _ in range(draws):
        ids_selected=rng.choice(images,len(images),replace=True)
        sample=[r for id in ids_selected for r in groups[id]]
        for key,val in stats(sample).items():
            if np.isfinite(val):boot[key].append(float(val))
    report={'dataset':'COCO 2017 real instance masks',
            'model':'Original Meta SAM ViT-B checkpoint, oracle GT box',
            'source_artifact_sha256':{str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'checkpoint_sha256':two[0][2]['checkpoint_sha256'],
            'n_images':len(images),'n_instances':len(rows),'n_grabcut_valid_pairs':len(paired),
            'no_image_overlap':True,'cluster_bootstrap_seed':seed,'draws':draws,
            'metric_summary':{k:{'estimate':float(v),
               'cluster_95_interval':[float(x) for x in np.percentile(boot[k],[2.5,97.5])]}
               for k,v in observed.items()},
            'per_source':{label:{'n_images':len(p['image_ids']),'n_instances':len(s),
               'sam_mean_iou':float(np.mean([r['sam_iou'] for r in s]))}
               for label,(s,p,m) in zip(('seeded_18','sorted_24'),two)},
            'limitations':'Conditional image-cluster bootstrap of two NONRANDOM samples; no full COCO official AP, no prompt-independent performance or external generalization; GrabCut pairing is filtered by rectangle validity.'}
    out.mkdir(parents=True,exist_ok=True)
    (out/'sam_42_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    with (out/'sam_42_all_instances.csv').open('w',newline='') as f:
        cols=['experiment','image_id','annotation_id','class','ground_truth_area_px','sam_iou','sam_dice','grabcut_iou','grabcut_dice','sam_predicted_iou','grabcut_valid'];w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for r in rows:w.writerow({k:r[k] for k in cols})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(10.6,3.6),facecolor='white')
    src=['Pilot 3 images','Seeded 18','Sorted 24','Pooled 42']
    mean_pilot=.8628225174975672
    means=[mean_pilot,report['per_source']['seeded_18']['sam_mean_iou'],report['per_source']['sorted_24']['sam_mean_iou'],float(observed['sam_mean_iou'])]
    axs[0].plot(range(4),means,'o',markersize=6,color='#178078')
    axs[0].errorbar([3],[means[-1]],yerr=[[means[-1]-report['metric_summary']['sam_mean_iou']['cluster_95_interval'][0]],[report['metric_summary']['sam_mean_iou']['cluster_95_interval'][1]-means[-1]]],fmt='none',capsize=4,color='#c2693c')
    axs[0].set_xticks(range(4),src,rotation=16,ha='right');axs[0].set_ylim(0,1)
    axs[0].set_ylabel('Object-wise mask IoU')
    axs[0].grid(axis='y',alpha=.2)
    sns=['SAM, paired','GrabCut, paired','Paired difference']
    keys=['paired_sam_iou','paired_grabcut_iou','paired_difference']
    for y,(label,key) in enumerate(zip(sns,keys)):
        data=report['metric_summary'][key];l,h=data['cluster_95_interval']
        axs[1].plot([l,h],[y,y],lw=2,color='#178078')
        axs[1].plot(data['estimate'],y,'o',color='#c2693c',ms=6)
    axs[1].set_yticks(range(3),sns);axs[1].set_xlim(0,1);axs[1].invert_yaxis()
    axs[1].set_xlabel('Paired object-wise mask IoU / difference')
    axs[1].grid(axis='x',alpha=.2)
    for j,ax in enumerate(axs):ax.spines[['right','top']].set_visible(False);ax.text(.5,-.35,f'({"ab"[j]})',transform=ax.transAxes,ha='center')
    fig.subplots_adjust(left=.11,right=.98,top=.96,bottom=.32,wspace=.5)
    for fmt in ('pdf','png','svg'):fig.savefig(out/f'sam_42_conditional_uncertainty.{fmt}',dpi=200,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    return report

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('seeded18',type=Path);a.add_argument('sorted24',type=Path);a.add_argument('--out',type=Path,required=True)
    args=a.parse_args();r=analyze([args.seeded18,args.sorted24],args.out)
    print(json.dumps({k:v for k,v in r.items() if k not in ('source_artifact_sha256','limitations')},indent=2))
