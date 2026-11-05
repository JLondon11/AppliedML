"""Scene-cluster bootstrap on actual SAM COCO per-object mask IoU, not COCO AP."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np

def bootstrap(csv_path,draws=5000,seed=20261104):
    if draws<200:raise ValueError('Need >=200 draws')
    groups=defaultdict(list)
    with open(csv_path,newline='') as f:
        for r in csv.DictReader(f):
            if not {'image_id','sam_iou','grabcut_iou','grabcut_valid'}.issubset(r):
                raise ValueError('Missing required columns')
            s=float(r['sam_iou'])
            paired=r['grabcut_valid'].lower()=='true'
            g=float(r['grabcut_iou']) if paired else np.nan
            if not 0<=s<=1 or (paired and not 0<=g<=1):
                raise ValueError('Invalid genuine mask IoU')
            groups[r['image_id']].append((s,g,paired))
    keys=list(groups)
    if len(keys)<5:raise ValueError('Need >=5 independent COCO images for scene-level bootstrap')
    def metrics(selected):
        objects=[t for key in selected for t in groups[key]]
        paired=[t for t in objects if t[2]]
        if not paired:return [float(np.mean([t[0] for t in objects])),np.nan,np.nan,np.nan]
        return [float(np.mean([t[0] for t in objects])),
                float(np.mean([t[0] for t in paired])),
                float(np.mean([t[1] for t in paired])),
                float(np.mean([t[0]-t[1] for t in paired]))]
    point=metrics(keys)
    rng=np.random.default_rng(seed)
    values=np.asarray([metrics(rng.choice(keys,size=len(keys),replace=True))
                       for _ in range(draws)],dtype=float)
    names=('sam_mean_iou','paired_sam_iou','paired_grabcut_iou','paired_iou_difference')
    stats={}
    for j,k in enumerate(names):
        clean=values[:,j][np.isfinite(values[:,j])]
        if len(clean)<draws*.90:raise ValueError('Too few paired bootstrap samples')
        stats[k]={'point':point[j],'lower95':float(np.quantile(clean,.025)),
                  'upper95':float(np.quantile(clean,.975))}
    return {'n_images':len(keys),'n_objects':sum(map(len,groups.values())),
      'n_paired':sum(int(t[2]) for values in groups.values() for t in values),
      'draws':draws,'seed':seed,'metrics':stats,
      'warning':'Exploratory 95% interval conditional on selected COCO scenes; not official COCO AP or general population performance.'}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--per-object-csv',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--draws',type=int,default=5000)
    p.add_argument('--seed',type=int,default=20261104)
    a=p.parse_args()
    data=bootstrap(a.per_object_csv,a.draws,a.seed)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'bootstrap_uncertainty.json').write_text(json.dumps(data,indent=2)+'\n')
    with (a.output/'bootstrap_uncertainty.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['metric','mean','lower95','upper95'])
        for k,v in data['metrics'].items():w.writerow([k,v['point'],v['lower95'],v['upper95']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(data['metrics'])
    fig,ax=plt.subplots(figsize=(7.2,3.2),facecolor='white')
    for j,k in enumerate(names):
        v=data['metrics'][k];mean=v['point']
        ax.errorbar(mean,3-j,xerr=[[mean-v['lower95']],[v['upper95']-mean]],
                    fmt='o',capsize=3,ms=5,color='#087f7b')
    ax.set_yticks(range(4),names[::-1])
    ax.set_xlabel('Mean IoU and image-cluster bootstrap interval (95%)')
    ax.axvline(0,color='#9099a8',ls=':')
    ax.spines[['right','top']].set_visible(False)
    fig.tight_layout()
    for ext in ('png','pdf','svg'):
        fig.savefig(a.output/f'sam_uncertainty.{ext}',dpi=200,facecolor='white',bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(data,indent=2))
if __name__=='__main__':main()
