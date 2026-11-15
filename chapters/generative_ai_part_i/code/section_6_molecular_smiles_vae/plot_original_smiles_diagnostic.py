"""Scientific SMILES-VAE diagnostics on original NCI molecule outputs.

Requires actual saved 600 original molecule identities, held-out decoded strings
and 1,024 prior samples from run.py. Rejects missing original observations.
No invented medicinal activity, chemical synthesis or external FID scores.
"""
from pathlib import Path
import argparse,csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def records(folder,name):
    with (folder/name).open(newline='',encoding='utf-8') as stream:
        return list(csv.DictReader(stream))

def main(input_dir,output_dir):
    input_dir=Path(input_dir);output_dir=Path(output_dir)
    manifest=records(input_dir,'source_manifest.csv')
    heldout=records(input_dir,'heldout_reconstructions.csv')
    prior=records(input_dir,'prior_samples.csv')
    metrics=json.loads((input_dir/'metrics.json').read_text())
    if len(manifest)!=600 or len(heldout)!=120 or len(prior)!=1024:
        raise ValueError('Original real NCI source/output rows incomplete')
    if len({r['canonical_smiles'] for r in manifest})!=600:
        raise ValueError('Source molecular structures are not unique')
    if len({r['source_id'] for r in heldout})!=120:
        raise ValueError('Original held-out molecule identities duplicated')
    valid_test=np.array([int(r['valid']) for r in heldout])
    exact=np.array([int(r['exact_reconstruction']) for r in heldout])
    sim=np.array([float(r['morgan_tanimoto']) for r in heldout])
    valid_prior=np.array([int(r['valid']) for r in prior])
    if not np.all((sim>=0)&(sim<=1)):raise ValueError('Invalid original fingerprint metrics')
    if valid_test.sum()!=69 or exact.sum()!=0 or valid_prior.sum()!=36:
        raise ValueError('Original negative validity/reconstruction totals mismatch')
    if len({r['canonical_smiles'] for r in prior if int(r['valid'])})!=23:
        raise ValueError('Original valid canonical strings mismatch')
    if abs(sim.mean()-.05407)>.0001:raise ValueError('Original held-out Morgan similarity mismatch')
    if abs(valid_prior.mean()-metrics['prior_validity'])>1e-12:
        raise ValueError('Original prior validity source mismatch')
    output_dir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(10.4,5.0))
    navy='#23445b';teal='#1e7b80';copper='#bb6d50'
    values=np.sort(sim)
    axes[0].step(values,np.arange(1,len(values)+1)/len(values),
        where='post',color=teal,lw=2)
    axes[0].axvline(float(np.mean(sim)),color=copper,ls='--',lw=1.4)
    axes[0].set(xlim=(0,1),ylim=(0,1.01),xlabel='Test reconstruction Morgan fingerprint Tanimoto',
        ylabel='Empirical fraction of 120 original held-out molecules')
    axes[0].set_title('A   Held-out structural similarity',loc='left',weight='bold',color=navy)
    axes[0].text(.97,.12,f'Mean = {np.mean(sim):.3f}\nExact recovery = 0/120',
        transform=axes[0].transAxes,ha='right',fontsize=9,color=navy)
    categories=['Held-out decoded\nchemically valid','Unconditional prior\nchemically valid','Unique canonical\nvalid prior strings']
    totals=[len(heldout),len(prior),len(prior)]
    numerators=[int(valid_test.sum()),int(valid_prior.sum()),23]
    ys=[2,1,0]
    for y,num,den,color in zip(ys,numerators,totals,[teal,copper,navy]):
        axes[1].plot([0,100],[y,y],lw=6,color='#e8eef0',solid_capstyle='round')
        axes[1].plot([0,100*num/den],[y,y],lw=6,color=color,solid_capstyle='round')
        axes[1].scatter([100*num/den],[y],s=72,color=color,zorder=3)
        axes[1].text(min(100*num/den+2,78),y+.16,f'{num}/{den}  ({100*num/den:.1f}%)',
            fontsize=9,color=navy)
    axes[1].set(xlim=(0,100),ylim=(-.55,2.55),xlabel='Proportion of corresponding original output set (%)',
        yticks=ys,yticklabels=categories)
    axes[1].set_title('B   Actual molecular-string validity',loc='left',weight='bold',color=navy)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='x',color='#e8eef0')
    fig.suptitle('Original NCI molecular strings · observed VAE failure diagnostics',
        x=.055,y=.97,ha='left',color=navy,fontsize=14,weight='bold')
    fig.text(.055,.891,'600 genuine NCI molecules (360/120/120 train/validation/test); 1,024 newly generated numerical prior strings',
        fontsize=9,color='#526676')
    fig.text(.055,.027,'No biological endpoint, prospective synthesis, drug efficacy or general NCI validity is established.',
        fontsize=8.5,color='#526676')
    fig.subplots_adjust(left=.115,right=.98,top=.80,bottom=.20,wspace=.40)
    for ext in ('pdf','png','svg'):
        fig.savefig(output_dir/f'nci_smiles_original_failure_analysis.{ext}',
            dpi=240,facecolor='white')
    plt.close(fig)
    print(json.dumps({'original_source_molecules':600,'heldout_original':120,
        'test_structural_validity':int(valid_test.sum())/120,
        'exact_test_reconstructions':int(exact.sum()),
        'prior_samples':1024,'valid_prior_samples':int(valid_prior.sum()),
        'unique_valid_prior_canonical':23,
        'test_mean_tanimoto':float(sim.mean())},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.results,a.output)
