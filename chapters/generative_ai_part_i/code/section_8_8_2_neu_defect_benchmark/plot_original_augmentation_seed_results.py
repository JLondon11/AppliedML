"""Original real NEU-DET classification augmentation results by training seed.

Inputs are *actually executed* 3-seed original 360-real-image test records in
replicated_seed_metrics_pass13.csv, not generated placeholder measurements.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent

def main(out=HERE/'scientific_figure_results'):
    with (HERE/'replicated_seed_metrics_pass13.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    arms=['real_only','flip_augmentation','gan_augmented','real_repeat_augmented']
    seeds=[20261106,20261107,20261108]
    assert len(rows)==12
    assert {(int(r['seed']),r['arm']) for r in rows}=={(s,a) for s in seeds for a in arms}
    scores={(int(r['seed']),r['arm']):(float(r['test_accuracy']),float(r['test_macro_f1'])) for r in rows}
    for p in scores.values():
        assert all(0<=x<=1 for x in p)
    colors={'real_only':'#27637a','flip_augmentation':'#8796a4','gan_augmented':'#bb6b4b','real_repeat_augmented':'#137b72'}
    names={'real_only':'Real only','flip_augmentation':'Horizontal flips',
           'gan_augmented':'Conditional GAN','real_repeat_augmented':'Real repeats + flips'}
    delta=np.mean([scores[(s,'gan_augmented')][0]-scores[(s,'real_only')][0] for s in seeds])
    assert abs(delta-(-0.1925925925925926))<1e-8
    fig,axes=plt.subplots(1,2,figsize=(11.3,5.1))
    for ix,(ax,title) in enumerate(zip(axes,['Observed held-out classification accuracy','Observed held-out macro-F1'])):
        for arm in arms:
            measure=np.array([scores[(s,arm)][ix] for s in seeds])
            ax.plot(range(3),measure,color=colors[arm],lw=1.8,
                    marker='o',markersize=8,label=names[arm])
        ax.set_xticks([0,1,2],['Seed 1','Seed 2','Seed 3'])
        ax.set_ylim(.3,.9);ax.grid(axis='y',color='#dce6eb')
        ax.set_ylabel('Accuracy on 360 original real test images' if ix==0
                      else 'Macro-F1 on six real defect classes')
        ax.set_title(('A  ' if ix==0 else 'B  ')+title,loc='left',fontsize=11,fontweight='semibold')
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('NEU steel-surface defects  •  GAN augmentation did not improve classification',
        x=.054,y=.98,ha='left',fontsize=15,color='#1d3852',fontweight='bold')
    fig.text(.054,.895,'1,800 original photos, 1 exact duplicate excluded. 1,080 train / 359 validation / 360 held-out test.',
        fontsize=9.3,color='#526676')
    fig.text(.054,.850,f'Same real test images, three training seeds. GAN minus real-only accuracy: {delta*100:+.2f} percentage points.',
        fontsize=9.3,color='#526676')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,
               frameon=False,bbox_to_anchor=(.5,.10),fontsize=9)
    fig.text(.054,.018,'Measured Pass 13 seed scores; no independent GAN-image fidelity or population-level generalization established.',
        color='#526676',fontsize=8.5)
    fig.subplots_adjust(left=.07,right=.98,top=.76,bottom=.22,wspace=.25)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for ext in ('pdf','png','svg'):
        fig.savefig(out/f'neu_original_real_seed_comparison.{ext}',dpi=235,facecolor='white')
    plt.close(fig)
    print('Measured GAN minus real-only accuracy, mean over seeds:',delta)

if __name__=='__main__':main()
