"""Scientific study-design diagram from authentic executed NEU-DET metrics.

No unseen photographs are fabricated; abstract rectangles depict data partitions.
The four measured test outcomes are read only from actual Pass13 seed data.
"""
from pathlib import Path
from collections import defaultdict
import csv, json, hashlib, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C='#173f59'; D='#526b7a'; A='#13787c'; O='#ad684c'
EXPECTED=['real_only','flip_augmentation','gan_augmented','real_repeat_augmented']
SEEDS=[20261106,20261107,20261108]

def verify(records):
    with Path(records).open(newline='', encoding='utf8') as f:
        data=list(csv.DictReader(f))
    assert len(data)==12
    assert {(int(r['seed']),r['arm']) for r in data}=={(s,a) for s in SEEDS for a in EXPECTED}
    scores=defaultdict(dict)
    for r in data:
        assert 0<=float(r['test_accuracy'])<=1 and 0<=float(r['test_macro_f1'])<=1
        scores[r['arm']][int(r['seed'])]=(float(r['test_accuracy']),float(r['test_macro_f1']))
    means={arm:tuple(sum(scores[arm][seed][k] for seed in SEEDS)/3 for k in (0,1)) for arm in EXPECTED}
    assert abs(means['gan_augmented'][0]-0.5268518519)<1e-9
    assert abs(means['real_only'][0]-0.7194444444)<1e-9
    assert abs(means['real_repeat_augmented'][0]-0.7962962963)<1e-9
    return data,means

def pill(ax,x,y,w,h,title,subtitle,outline=A):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012,rounding_size=.018',
        ec=outline,fc='white',lw=1.1,transform=ax.transAxes))
    ax.text(x+.018,y+h*.64,title,transform=ax.transAxes,fontsize=9.2,weight='bold',color=C,va='center')
    ax.text(x+.018,y+h*.26,subtitle,transform=ax.transAxes,fontsize=8.15,color=D,va='center')

def make(source,out):
    data,means=verify(source)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(6.5,8.3),facecolor='white');ax.set_axis_off()
    fig.text(.058,.966,'Real steel-surface defects',fontsize=15,weight='bold',color=C)
    fig.text(.058,.935,'Actual NEU-DET classification and augmentation protocol',fontsize=9.1,color=D)
    pill(ax,.07,.810,.86,.099,'Authentic source photography',
        '1,800 grayscale defect images; 6 labeled surface classes')
    pill(ax,.07,.682,.86,.095,'Source partitions and duplicate control',
        '1,080 train / 359 validation / 360 real test; 1 duplicate excluded')
    ax.add_patch(FancyArrowPatch((.5,.81),(.5,.787),arrowstyle='-|>',mutation_scale=12,
         color=D,transform=ax.transAxes))
    pill(ax,.07,.550,.86,.097,'Matched original-test evaluation',
        '4 training arms x 3 seeds; SAME 360 real held-out photos')
    ax.add_patch(FancyArrowPatch((.5,.682),(.5,.655),arrowstyle='-|>',mutation_scale=12,
         color=D,transform=ax.transAxes))
    fig.text(.073,.508,'Original test results across three executed seeds',fontsize=10,weight='bold',color=C)
    fig.text(.073,.478,'Training regime',fontsize=8.3,color=D)
    fig.text(.687,.478,'Accuracy',fontsize=8.3,color=D)
    fig.text(.851,.478,'Macro-F1',fontsize=8.3,color=D)
    names={
      'real_only':'Real originals only',
      'flip_augmentation':'Horizontal flip augmentation',
      'gan_augmented':'Conditional GAN + originals',
      'real_repeat_augmented':'Real repeats + flips (matched count)'}
    for j,key in enumerate(EXPECTED):
        y=.426-j*.057
        rect=FancyBboxPatch((.065,y-.013),.87,.052,
             boxstyle='round,pad=0.004,rounding_size=.006',
             ec='#e0e8ed',fc=('#fff3ee' if key=='gan_augmented' else '#f5f9fa'),
             lw=.65,transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(.084,y+.012,names[key],fontsize=8.8,transform=ax.transAxes,color=C,va='center',weight='bold' if key=='gan_augmented' else 'normal')
        ax.text(.686,y+.012,f'{100*means[key][0]:.2f}%',fontsize=9.0,transform=ax.transAxes,color=O if key=='gan_augmented' else C,va='center')
        ax.text(.852,y+.012,f'{100*means[key][1]:.2f}%',fontsize=9.0,transform=ax.transAxes,color=O if key=='gan_augmented' else C,va='center')
    fig.text(.073,.186,'Measured contrast (GAN minus original-only): -19.26 pp',fontsize=9.6,color=O,weight='bold')
    fig.text(.073,.158,'GAN-trained samples are model outputs, not original defect photographs.',fontsize=8.1,color=D)
    fig.text(.073,.127,'Short GAN training (300 steps/seed); 12 classifier epochs/seed.',fontsize=8.1,color=D)
    fig.text(.073,.098,'This is an executed offline classification study, not plant deployment.',fontsize=8.1,color=D)
    fig.text(.073,.066,'No image-quality, industrial reliability, or external-factory claim is made.',fontsize=8.1,color=D)
    fig.text(.073,.035,'Diagram shows original measured data flow; boxes are schematic, not sample images.',fontsize=8.1,color=D)
    fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
    for fmt in ('pdf','svg','png'):
        fig.savefig(out/f'neu_real_experiment_protocol.{fmt}',dpi=230,facecolor='white')
    plt.close(fig)
    manifest={
        'source':'Authentic executed NEU-DET Pass13 three-seed metrics',
        'source_csv_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        'original_NEU_dataset_archive_sha256':'2ccaff85d88567c52708c2f9dc4dec10cff8650749df2984f03b38716998460f',
        'original_images':1800,'true_train':1080,'true_validation':359,'true_test':360,
        'training_seeds':SEEDS,'measured_mean_scores':{k:{'accuracy':v[0],'macro_f1':v[1]} for k,v in means.items()},
        'plotted_original_photos':0,'note':'Boxes are scientific experiment schematic; no synthetic/fabricated photos or laboratory outcomes',
        'manuscript_figure_620_identified':False,'production_approval':False}
    (out/'neu_scientific_diagram_provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(make(a.source,a.output),indent=2))
