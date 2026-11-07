"""Generate scientific test-result figures from the four-arm real NEU benchmark."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,recall_score

ROOT=Path(__file__).resolve().parent/'results_multiseed'
FIG=ROOT/'figures'

def main():
    FIG.mkdir(exist_ok=True,parents=True)
    scores=pd.concat((pd.read_csv(ROOT/'benchmark_metrics.csv'),
                      pd.read_csv(ROOT/'repeat_control_metrics.csv')))
    predictions=pd.concat((pd.read_csv(ROOT/'test_predictions.csv'),
                           pd.read_csv(ROOT/'repeat_control_predictions.csv')))
    arms=['real_only','flip_augmentation','gan_augmented','real_repeat_augmented']
    labels=['Real only','Horizontal flips','GAN augmented','Real repeat + flips']
    colors=['#1F4E79','#7B9EAA','#B25D4C','#47875C']
    fig,ax=plt.subplots(figsize=(9.3,4.55))
    for i,(arm,color) in enumerate(zip(arms,colors)):
        v=scores[scores.arm==arm].sort_values('seed').test_balanced_accuracy.to_numpy()
        ax.scatter(i+np.array([-.11,0,.11]),v,color=color,s=54,zorder=3,
                   edgecolor='white',linewidth=.6)
        ax.errorbar(i,v.mean(),yerr=v.std(ddof=1),fmt='D',markersize=6,
                    capsize=5,lw=1.4,color=color,zorder=4)
        ax.text(i,v.mean()+.063,f'{100*v.mean():.1f}%',ha='center',fontsize=10)
    ax.set(xticks=range(4),xticklabels=labels,ylabel='Held-out balanced accuracy',
           xlim=(-.5,3.5),ylim=(.32,.94))
    ax.grid(axis='y',alpha=.18)
    fig.tight_layout()
    fig.savefig(FIG/'four_arm_three_seed_accuracy.pdf')
    fig.savefig(FIG/'four_arm_three_seed_accuracy.png',dpi=220)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.6))
    classes=['Crazing','Inclusion','Patches','Pitted surface','Rolled-in scale','Scratches']
    for j,(arm,color) in enumerate(zip(arms,colors)):
        v=[]
        for cl in range(6):
            rec=[]
            for seed in sorted(scores.seed.unique()):
                q=predictions[(predictions.arm==arm)&(predictions.seed==seed)]
                rec.append(recall_score(q.true_class,q.pred_class,labels=[cl],
                                        average='macro',zero_division=0))
            v.append(np.mean(rec))
        ax.bar(np.arange(6)+(j-1.5)*.2,v,width=.19,label=labels[j],color=color)
    ax.set_xticks(range(6),classes,rotation=18,ha='right')
    ax.set(ylabel='Mean class recall across three seeds',ylim=(0,1))
    ax.legend(frameon=False,ncol=2,fontsize=8)
    ax.grid(axis='y',alpha=.18)
    fig.tight_layout()
    fig.savefig(FIG/'four_arm_classwise_recall.pdf')
    fig.savefig(FIG/'four_arm_classwise_recall.png',dpi=220)
    plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(11,9.8),constrained_layout=True)
    short=['Cr','In','Pa','Pi','Ro','Sc']
    for i,(arm,ax) in enumerate(zip(arms,axs.flat)):
        q=predictions[predictions.arm==arm]
        cm=confusion_matrix(q.true_class,q.pred_class,labels=range(6))
        ax.imshow(cm,cmap='Blues',vmin=0,vmax=180)
        ax.set_title(labels[i]);ax.set_xticks(range(6),short)
        ax.set_yticks(range(6),short)
        ax.set(xlabel='Predicted label',ylabel='Reference label')
        for (r,c),val in np.ndenumerate(cm):
            ax.text(c,r,str(val),ha='center',va='center',fontsize=8,
                    color='white' if val>90 else 'black')
    fig.savefig(FIG/'four_arm_confusion_matrices.pdf')
    fig.savefig(FIG/'four_arm_confusion_matrices.png',dpi=200)
    plt.close(fig)
    print('Finished publication figures')
if __name__=='__main__':main()
