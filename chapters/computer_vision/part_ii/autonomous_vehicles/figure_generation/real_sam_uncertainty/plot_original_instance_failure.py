"""Plot real original COCO per-instance SAM failures; no invented observations.

Use downloaded original GitHub Actions artifacts' per_object_metrics.csv.
Reports prompt-conditioned, selected-scene observations, not COCO AP.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read_records(source):
    with Path(source).open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    if len(set((r['image_id'],r['annotation_id']) for r in rows))!=len(rows):
        raise ValueError('Duplicate original annotation')
    return rows

def render(run18,run24,out):
    a,b=read_records(run18),read_records(run24)
    if len(a)!=59 or len(b)!=88:
        raise ValueError('Unexpected original experiment size')
    if set(r['image_id'] for r in a)&set(r['image_id'] for r in b):
        raise ValueError('Expected disjoint 18/24 original COCO scenes')
    rows=a+b
    area=np.array([float(r['ground_truth_area_px']) for r in rows])
    iou=np.array([float(r['sam_iou']) for r in rows])
    valid=np.array([r['grabcut_valid']=='True' for r in rows])
    difference=np.array([float(r['sam_iou'])-float(r['grabcut_iou']) for r in rows if r['grabcut_valid']=='True'])
    if len(difference)!=99 or not np.isfinite(np.r_[area,iou,difference]).all():
        raise ValueError('Incomplete original numerical records')
    if np.any(area<=0) or np.any((iou<0)|(iou>1)):
        raise ValueError('Invalid original area or IoU')
    if abs(float(iou.mean())-.7786187121390236)>1e-12:
        raise ValueError('Pooled original SAM mask IoU mismatch')
    if abs(float(difference.mean())-.3197954719125727)>1e-12:
        raise ValueError('Original paired IoU difference mismatch')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    navy='#173c56';teal='#147e82';copper='#ba6d4c'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axs=plt.subplots(1,2,figsize=(10.4,5.1),gridspec_kw={'width_ratios':[1.1,1]})
    for ax in axs:
        ax.grid(color='#e4ecef',lw=.75,alpha=.95);ax.set_axisbelow(True)
    axs[0].scatter(area[:59],iou[:59],s=22,color=teal,alpha=.75,
        edgecolor='white',linewidth=.35,label='18-scene SAM run')
    axs[0].scatter(area[59:],iou[59:],s=25,color=copper,marker='D',
        alpha=.75,edgecolor='white',linewidth=.35,label='24-scene SAM run')
    axs[0].axhline(iou.mean(),color=navy,lw=1.25,linestyle=(0,(4,3)),
        label=f'Pooled mean IoU = {iou.mean():.3f}')
    axs[0].set(xscale='log',xlabel='Original COCO annotation area (pixels²; log scale)',
        ylabel='Observed SAM mask IoU',ylim=(-.04,1.04),
        title='A  Size-related segmentation variation')
    axs[0].legend(loc='lower right',fontsize=8,frameon=False)
    axs[1].scatter(area[valid],difference,s=27,color=teal,alpha=.75,
        edgecolor='white',linewidth=.35)
    axs[1].axhline(0,color='#526578',lw=1.1)
    axs[1].axhline(difference.mean(),color=copper,lw=1.5,linestyle=(0,(4,3)),
        label=f'Mean Δ IoU = {difference.mean():+.3f}')
    axs[1].set(xscale='log',xlabel='Paired original annotation area (pixels²; log scale)',
        ylabel='Observed SAM − GrabCut mask IoU',ylim=(-1.03,1.03),
        title='B  Object-level paired error differences')
    axs[1].legend(loc='lower right',fontsize=8,frameon=False)
    fig.suptitle('Observed instance-level performance on original COCO images',
        fontsize=15,color=navy,x=.055,ha='left',y=.975,weight='bold')
    fig.subplots_adjust(left=.082,right=.987,top=.82,bottom=.245,wspace=.29)
    fig.text(.055,.126,'147 original SAM masks from 42 selected COCO photographs; 99 valid matched SAM–GrabCut comparisons.',
        fontsize=9,color='#425968')
    fig.text(.055,.083,'Frozen SAM ViT-B uses oracle GT box prompts. Area and IoU are measured original annotations/model outputs.',
        fontsize=8.5,color='#425968')
    fig.text(.055,.045,'Selected eligible scenes only; not representative COCO AP or autonomous-system performance.',
        fontsize=8.3,color='#425968')
    for ext in ('pdf','png','svg'):
        fig.savefig(out/f'sam_original_object_level_error.{ext}',dpi=250,facecolor='white')
    plt.close(fig)
    result={'original_images':42,'sam_objects':147,'matched_grabcut_objects':99,
        'sam_mean_iou':float(iou.mean()),
        'sam_objects_iou_below_0_5':int((iou<.5).sum()),
        'sam_better_paired':int((difference>0).sum()),
        'sam_equal_or_worse_paired':int((difference<=0).sum()),
        'paired_mean_iou_difference':float(difference.mean()),
        'not_covered':'Population confidence, prompt-free quality, autonomous driving and official COCO AP'}
    (out/'observed_failure_metrics.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--sam-18',type=Path,required=True)
    parser.add_argument('--sam-24',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    render(args.sam_18,args.sam_24,args.output)
