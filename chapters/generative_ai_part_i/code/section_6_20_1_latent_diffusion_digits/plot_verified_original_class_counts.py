"""Plot actual observed 2,000-sample digit model classifier results.

Default source is the archived Pass27 original class table (778/94). When a
different per-execution count table is provided, plot and report those actual
observations, without forcing the archived model's counts.
"""
import csv,argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(source,out,expected_conditional=None,expected_unconditional=None):
    with Path(source).open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    observed={(r['arm'],r['label']):(int(r['correct']),int(r['n'])) for r in rows}
    if len(rows)!=22 or len(observed)!=22:raise ValueError('Missing original model-count rows')
    arms=('conditional','unconditional')
    if set(observed)!={(arm,label) for arm in arms for label in ['overall',*map(str,range(10))]}:
        raise ValueError('Incorrect actual model arm or requested-digit inventory')
    totals={}
    for arm in arms:
        expected,n=observed[(arm,'overall')]
        if n!=1000 or not 0<=expected<=1000:
            raise ValueError('Current-run total 1000 generated samples not verified')
        if sum(observed[(arm,str(j))][0] for j in range(10))!=expected:
            raise ValueError('Current-run overall does not match measured class counts')
        if any(observed[(arm,str(j))][1]!=100 or not 0<=observed[(arm,str(j))][0]<=100
               for j in range(10)):
            raise ValueError('Expected 100 scored samples per actual digit class')
        totals[arm]=expected
    if expected_conditional is not None and totals['conditional']!=expected_conditional:
        raise ValueError('Archived conditional count mismatches its original source')
    if expected_unconditional is not None and totals['unconditional']!=expected_unconditional:
        raise ValueError('Archived unconditional count mismatches its original source')
    categories=np.arange(10)
    cond=np.array([observed[('conditional',str(j))][0] for j in categories])
    uncond=np.array([observed[('unconditional',str(j))][0] for j in categories])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10.8,5.9),facecolor='white')
    for d,a,b in zip(categories,cond,uncond):
        ax.plot([d,d],[b,a],color='#bcc9ce',lw=2,zorder=1)
    ax.scatter(categories,cond,color='#156a78',s=76,edgecolors='white',linewidths=.6,
       label='Class-conditioned generated digits',zorder=3)
    ax.scatter(categories,uncond,color='#b67051',s=71,marker='D',
       edgecolors='white',linewidths=.6,label='Unconditional generated digits',zorder=3)
    ax.axhline(10,color='#667e8a',lw=1.1,ls=(0,(3,3)),label='10% random-label reference')
    ax.set(xlim=(-.7,9.7),ylim=(-4,105),xticks=categories,yticks=range(0,101,20),
           xlabel='Requested digit class in original experiment',
           ylabel='Classifier agreement, observed count out of 100 (%)')
    ax.grid(axis='y',color='#e2eaee',linewidth=.9)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),frameon=False,ncol=3,fontsize=8.4)
    fig.text(.065,.95,'Original scanned-digit latent diffusion · observed per-class agreement',
       color='#193f5a',fontweight='bold',fontsize=15)
    fig.text(.065,.88,'2,000 actually generated samples scored by a classifier trained on real digit scans; 100 generated samples per arm and digit.',
       color='#4f6678',fontsize=9.2)
    fig.text(.065,.073,f"Observed totals: conditional {totals['conditional']}/1,000 ({totals['conditional']/10:.1f}%); unconditional {totals['unconditional']}/1,000 ({totals['unconditional']/10:.1f}%). One trained seed.",
       color='#193f5a',fontsize=9)
    fig.text(.065,.046,'Not Stable Diffusion, text-to-image, or an independent population-quality study. Dots are counts, not repeated training runs.',
       color='#516777',fontsize=8.4)
    fig.subplots_adjust(left=.09,right=.98,top=.77,bottom=.18)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for ext in ('png','svg'):
        fig.savefig(out/f'original_digit_classwise_scientific.{ext}',dpi=230,facecolor='white')
    plt.close(fig)
    print(f"PASS: current-model conditional {totals['conditional']}/1000, unconditional {totals['unconditional']}/1000; no archived count substitution")

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,
       default=Path(__file__).with_name('figure_6_44_class_counts_pass27.csv'))
    p.add_argument('--output',type=Path,default=Path(__file__).with_name('observed_figures'))
    p.add_argument('--expected-conditional',type=int,default=None,help='Optional original count for strict archive replay')
    p.add_argument('--expected-unconditional',type=int,default=None,help='Optional original count for strict archive replay')
    a=p.parse_args();main(a.source,a.output,a.expected_conditional,a.expected_unconditional)
