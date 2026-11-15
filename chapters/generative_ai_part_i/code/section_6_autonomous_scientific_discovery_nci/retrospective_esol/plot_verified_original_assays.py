"""Publication-quality 226-authentic-assay ESOL retrospective comparison.

NO newly measured chemistry; uses the independently recomputed original-source
test rows in verified_original_ESOL_226_test_predictions.csv. The scores
are genuine Delaney original measurements and two matched existing predictors.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent

def main(out):
    with (ROOT/'verified_original_ESOL_226_test_predictions.csv').open(newline='',encoding='utf-8') as fh:
        records=list(csv.DictReader(fh))
    assert len(records)==226 and len(set(r['smiles'] for r in records))==226
    y=np.array([float(r['measured_logS']) for r in records])
    a=np.array([float(r['ridge']) for r in records])
    b=np.array([float(r['published_ESOL']) for r in records])
    hits=np.array([int(r['hit']) for r in records])
    threshold=-1.6
    assert np.array_equal(hits,(y>=threshold).astype(int))
    sorted_a=np.argsort(-a)[:20];sorted_b=np.argsort(-b)[:20]
    n_a=int(hits[sorted_a].sum());n_b=int(hits[sorted_b].sum())
    rmse_a=float(np.sqrt(np.mean((y-a)**2)));rmse_b=float(np.sqrt(np.mean((y-b)**2)))
    assert n_a==14 and n_b==17
    assert abs(rmse_a-1.1795769332543176)<1e-9
    assert abs(rmse_b-.8755685030537721)<1e-9
    navy='#183d57';teal='#147c80';copper='#b56c48'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,
        'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10.4,5.45))
    bounds=[min(y.min(),a.min(),b.min())-.4,max(y.max(),a.max(),b.max())+.4]
    for ax,pred,mark,color,title,rmse,ids,n in zip(
        axes,[a,b],['o','D'],[teal,copper],
        ['A   Locally fitted ridge','B   Published ESOL predictor'],
        [rmse_a,rmse_b],[sorted_a,sorted_b],[n_a,n_b]):
        ax.scatter(y,pred,s=23,color=color,alpha=.5,edgecolor='white',lw=.25)
        ax.scatter(y[ids],pred[ids],s=57,marker=mark,facecolor='none',
            edgecolor=navy,lw=1.15,label='Top 20 predicted compounds')
        ax.plot(bounds,bounds,ls='--',color='#8a9ca9',lw=1.2,
                label='Perfect prediction')
        ax.axvline(threshold,ls=':',color='#748b99',lw=1)
        ax.set(xlim=bounds,ylim=bounds,
            xlabel='Original measured log solubility (log mol/L)',
            ylabel='Model-predicted log solubility (log mol/L)')
        ax.set_title(title,loc='left',fontsize=11,color=navy,weight='bold')
        ax.text(.035,.96,f'RMSE {rmse:.3f}  ·  top-20 hits {n}/20',
            transform=ax.transAxes,va='top',color=navy,fontsize=9,
            bbox={'facecolor':'white','edgecolor':'#e5eaef','alpha':.92,'pad':4})
        ax.grid(color='#e7ecef',alpha=.85)
    fig.suptitle('Measured ESOL solubility · retrospective molecular ranking',
        x=.048,y=.975,ha='left',fontsize=15.5,color=navy,weight='bold')
    fig.text(.048,.902,'1,128 existing original assays · 902 training / 226 held-out · threshold ≥ −1.6 log mol/L (training 75th percentile)',
        fontsize=9,color='#526676')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,
        bbox_to_anchor=(.5,.072),frameon=False,fontsize=9)
    fig.text(.048,.027,'One retrospective random split; no prospectively assayed molecules, generated drugs or autonomous laboratory experiment.',
        fontsize=8.3,color='#526676')
    fig.subplots_adjust(left=.09,right=.985,top=.80,bottom=.20,wspace=.25)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for fmt in ('pdf','png','svg'):
        fig.savefig(out/f'esol_original_assay_retrospective.{fmt}',dpi=240,facecolor='white')
    plt.close(fig)
    report={'n_original_heldout':len(records),'ridge_rmse':rmse_a,
      'published_esol_rmse':rmse_b,'threshold':threshold,
      'ridge_top20_measured_hits':n_a,'published_esol_top20_measured_hits':n_b,
      'original_measured_test_positives':int(hits.sum()),
      'independent_test_molecules_not_generated':True}
    (out/'verified_original_figure_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    main(args.output)
