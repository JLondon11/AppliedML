"""Measured-only SAM uncertainty plot. Reads actual pooled original-COCO SAM JSON.
No new inference or fabricated samples; figure intervals come from original run.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

KEYS=('sam_mean_iou','sam_mean_dice','paired_sam_iou','paired_grabcut_iou','paired_difference')

def validate(data):
    if (data.get('distinct_images'),data.get('total_objects'),data.get('paired_grabcut_valid_objects'))!=(42,147,99):
        raise ValueError('Expected source experiment counts have changed: re-audit before publishing')
    if data['bootstrap']['clusters']!='image IDs':raise ValueError('Wrong bootstrap clustering')
    r=data['results']
    for key in KEYS:
        q=r[key]
        lo,mu,hi=(q['lower95'],q['estimate'],q['upper95'])
        if not np.isfinite([lo,mu,hi]).all() or not (0<=lo<=mu<=hi<=1):
            raise ValueError(f'Invalid original interval for {key}')
    delta=r['paired_sam_iou']['estimate']-r['paired_grabcut_iou']['estimate']
    if abs(delta-r['paired_difference']['estimate'])>1e-12:
        raise ValueError('Paired original-object difference cannot be reconciled')
    return r

def render(source, output):
    source=Path(source);output=Path(output)
    data=json.loads(source.read_text(encoding='utf-8'))
    r=validate(data)
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
                         'pdf.fonttype':42,'savefig.facecolor':'white'})
    navy='#193E57';teal='#117C80';rust='#B56C48';gray='#4A6270'
    fig=plt.figure(figsize=(10,5.45))
    gs=fig.add_gridspec(1,2,width_ratios=[1,1.03],left=.12,right=.955,bottom=.205,top=.79,wspace=.37)
    a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1])
    def mark(ax,k,y,color):
        q=r[k];lo=q['lower95'];mu=q['estimate'];hi=q['upper95']
        ax.plot([lo,hi],[y,y],color=color,lw=3,solid_capstyle='round',zorder=2)
        for x in (lo,hi):ax.plot([x,x],[y-.075,y+.075],color=color,lw=1.3)
        ax.scatter(mu,y,s=78,color=color,edgecolor='white',lw=1.2,zorder=3)
        ax.text(hi+.014,y,f'{mu:.3f}',color=navy,va='center',fontsize=9)
    mark(a,'sam_mean_iou',1,teal);mark(a,'sam_mean_dice',0,navy)
    a.set(yticks=[1,0],yticklabels=['Mask IoU','Mask Dice'],xlim=(.65,.98),ylim=(-.55,1.55),
          xlabel='Metric (0–1)',title='A  Original SAM masks  ·  147 objects')
    mark(b,'paired_sam_iou',2,teal);mark(b,'paired_grabcut_iou',1,rust)
    mark(b,'paired_difference',0,navy)
    b.set(yticks=[2,1,0],yticklabels=['SAM IoU','GrabCut IoU','SAM − GrabCut'],xlim=(.15,1),ylim=(-.55,2.55),
          xlabel='IoU or paired difference',title='B  Matched boxes  ·  99 objects')
    for ax in (a,b):
        ax.set_axisbelow(True);ax.grid(axis='x',lw=.8,color='#DFE6E9')
        ax.spines[['left','top','right']].set_visible(False)
        ax.tick_params(axis='y',length=0,pad=10);ax.tick_params(axis='x',length=0,pad=5)
    fig.text(.055,.935,'SAM ViT-B on genuine COCO photographs',color=navy,fontsize=17,weight='bold')
    fig.text(.055,.875,'Estimated mask quality and scene-cluster uncertainty  |  42 real images',color=gray,fontsize=10.5)
    fig.text(.055,.125,'Dots: observed means    |    Lines: exploratory 95% image-cluster bootstrap intervals (10,000 draws)',color=gray,fontsize=9.6)
    fig.text(.055,.077,'Oracle ground-truth box prompts supplied to SAM; selected eligible scenes, not official COCO AP.',color=gray,fontsize=9)
    fig.text(.055,.044,'The paired SAM–GrabCut difference uses only 99 valid matched objects; intervals are conditional, not population claims.',color=gray,fontsize=8.5)
    for ext in ('png','pdf','svg'):
        fig.savefig(output/f'sam_coco_measured_uncertainty.{ext}',dpi=250,facecolor='white')
    plt.close(fig)
    evidence={'original_images':42,'original_instances':147,'paired_instances':99,
              'checkpoint_sha256':data['checkpoint_sha256'],
              'original_input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'results':r,'scope':data['warning']}
    (output/'figure_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();render(a.source,a.output)
