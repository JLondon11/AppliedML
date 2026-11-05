#!/usr/bin/env python3
"""Paired real-mask evidence from successful pretrained SAM and Mask R-CNN runs.

Exact 18-image COCO IDs and 59 COCO annotation IDs are validated. Differences
are informative but NOT same-input fairness: SAM receives oracle GT box prompts.
"""
import argparse,csv,io,json,zipfile,hashlib
from pathlib import Path
import numpy as np

def parse(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        rows=list(csv.DictReader(io.StringIO(z.read('per_object_metrics.csv').decode())))
        manifest=json.loads(z.read('metrics_summary.json'))
    return {(int(x['image_id']),int(x['annotation_id'])):x for x in rows},manifest

def run(sam_zip,mask_zip,out,seed=20261105,draws=10000):
    a,ma=parse(sam_zip);b,mb=parse(mask_zip)
    if set(a)!=set(b):raise ValueError('SAM and Mask R-CNN used different annotated instances')
    if ma['images']!=mb['images']:raise ValueError('Different number of original COCO scenes')
    rows=[]
    for k in sorted(a):
        x=a[k];y=b[k]
        if x['class']!=y['class']:raise ValueError('Category mismatch')
        rows.append({'image_id':k[0],'annotation_id':k[1],'class':x['class'],
                     'sam_iou':float(x['sam_iou']), 'maskrcnn_iou':float(y['mask_iou']),
                     'difference':float(x['sam_iou'])-float(y['mask_iou']),
                     'maskrcnn_matched':y['matched'].lower()=='true'})
    groups={image:[r for r in rows if r['image_id']==image] for image in sorted(set(r['image_id'] for r in rows))}
    observed=float(np.mean([r['difference'] for r in rows]));rng=np.random.default_rng(seed)
    ids=np.array(sorted(groups));replicates=[]
    for _ in range(draws):
        selected=rng.choice(ids,len(ids),replace=True)
        rs=[r for id in selected for r in groups[int(id)]]
        replicates.append(float(np.mean([r['difference'] for r in rs])))
    ci=[float(x) for x in np.percentile(replicates,[2.5,97.5])]
    out.mkdir(parents=True,exist_ok=True)
    with (out/'sam_maskrcnn_matched_instances.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    stats={'n_original_coco_images':len(ids),'n_exact_paired_coco_instances':len(rows),
           'n_maskrcnn_matched':int(sum(r['maskrcnn_matched'] for r in rows)),
           'sam_mean_oraclebox_iou':float(np.mean([r['sam_iou'] for r in rows])),
           'maskrcnn_mean_automatic_matched_iou':float(np.mean([r['maskrcnn_iou'] for r in rows])),
           'mean_sam_minus_maskrcnn_iou':observed,'image_cluster_95_interval':ci,
           'bootstrap_draws':draws,'bootstrap_seed':seed,
           'sam_checkpoint_sha256':ma['checkpoint_sha256'],
           'maskrcnn_checkpoint_sha256':mb['checkpoint_sha256'],
           'sam_zip_sha256':hashlib.sha256(sam_zip.read_bytes()).hexdigest(),
           'maskrcnn_zip_sha256':hashlib.sha256(mask_zip.read_bytes()).hexdigest(),
           'scope':'Same 59 original COCO annotated objects; SAM gets oracle ground-truth box prompts and Mask R-CNN does not. Difference is NOT an equal-information or official COCO AP comparison; exploratory selected-scene bootstrap.'}
    (out/'sam_maskrcnn_comparison.json').write_text(json.dumps(stats,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10.3,3.8),facecolor='white')
    x=np.array([r['maskrcnn_iou'] for r in rows]);y=np.array([r['sam_iou'] for r in rows])
    axes[0].scatter(x,y,c='#15817b',alpha=.65,s=18)
    axes[0].plot([0,1],[0,1],ls='--',c='#888888',lw=1)
    axes[0].set(xlim=(0,1),ylim=(0,1),xlabel='Mask R-CNN matched-instance IoU',
                ylabel='SAM oracle-box mask IoU')
    axes[1].hist(replicates,bins=40,color='#668d97',edgecolor='white',linewidth=.3)
    axes[1].axvline(observed,color='#bc693d',lw=2)
    axes[1].axvline(ci[0],color='#344b63',ls='--',lw=1)
    axes[1].axvline(ci[1],color='#344b63',ls='--',lw=1)
    axes[1].set(xlabel='Image-cluster mean (SAM − Mask R-CNN) IoU',ylabel='Resamples')
    for i,ax in enumerate(axes):
        ax.spines[['top','right']].set_visible(False)
        ax.text(.5,-.3,'('+'ab'[i]+')',ha='center',transform=ax.transAxes)
    fig.subplots_adjust(left=.095,right=.985,bottom=.3,top=.97,wspace=.34)
    for ext in ('pdf','png','svg'):
        fig.savefig(out/f'sam_maskrcnn_59_comparison.{ext}',dpi=200,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    return stats

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('sam18',type=Path);p.add_argument('maskrcnn18',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.sam18,a.maskrcnn18,a.out),indent=2))
