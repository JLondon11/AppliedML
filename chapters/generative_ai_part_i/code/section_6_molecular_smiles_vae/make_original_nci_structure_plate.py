"""Image-first genuine NCI molecule/source-partition plate, not generated chemistry.

RDKit 2D bond diagrams are computed from original first_5K.smi real NCI
source structures, not chemical potency assays or VAE-generated compounds.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from rdkit import Chem,RDConfig
from rdkit.Chem import Draw,rdMolDescriptors
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
def main(out=OUT):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    source=Path(RDConfig.RDDataDir)/'NCI'/'first_5K.smi'
    source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    if source_sha!='91e71c015f14939837f2943dcc904f7c87e5a3a0124d82b05c28ad2f23004def':
        raise ValueError('Original NCI source file identity differs')
    smiles=[];identifiers=[];seen=set()
    for line in source.read_text().splitlines():
        tokens=line.strip().split()
        if len(tokens)<2:continue
        mol=Chem.MolFromSmiles(tokens[0])
        if mol is None:continue
        s=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=False)
        if len(s)>75 or s in seen:continue
        seen.add(s);smiles.append(s);identifiers.append(tokens[1])
        if len(smiles)==600:break
    if len(smiles)!=600:raise ValueError('Expected exact 600 real source molecules')
    ids=np.arange(600)
    train,test=train_test_split(ids,test_size=.2,random_state=20261109)
    train,val=train_test_split(train,test_size=.25,random_state=20261109)
    assert (len(train),len(val),len(test))==(360,120,120)
    chosen=[]
    for split,source_indices in (('Training',train),('Validation',val),('Test',test)):
        arr=sorted(source_indices.tolist())
        for index in (arr[1],arr[len(arr)//3],arr[2*len(arr)//3],arr[-2]):
            chosen.append((split,index))
    fig,axes=plt.subplots(3,4,figsize=(12.1,10),facecolor='white')
    records=[]
    for j,(split,index) in enumerate(chosen):
        mol=Chem.MolFromSmiles(smiles[index])
        row,col=divmod(j,4)
        ax=axes[row,col]
        ax.imshow(Draw.MolToImage(mol,size=(400,290),kekulize=True))
        ax.set_xticks([]);ax.set_yticks([])
        ax.set_title(f'{split} · NCI {identifiers[index]}',loc='left',
            fontsize=10,weight='bold',color='#163e56')
        formula=rdMolDescriptors.CalcMolFormula(mol)
        ax.text(.03,.03,formula,transform=ax.transAxes,fontsize=9,color='#3c5567',
            bbox={'facecolor':'white','alpha':.94,'boxstyle':'round,pad=.25',
                  'edgecolor':'#e0e8ed'})
        records.append({'split':split,'source_position':index,
            'nci_id':identifiers[index],'canonical_smiles':smiles[index],
            'rdkit_computed_molecular_formula':formula})
    for ax in axes.flat:
        for spine in ax.spines.values():spine.set_edgecolor('#dbe4e8')
    fig.suptitle('Authentic NCI molecules · observed chemical graph diversity',
        x=.04,y=.984,ha='left',fontsize=16,color='#173f58',weight='bold')
    fig.text(.04,.94,'Twelve original RDKit NCI source records, drawn as chemical structures. Four examples per genuine molecular train/validation/test partition.',
        fontsize=10,color='#536777')
    fig.text(.04,.035,'2D bond geometry and molecular formulas are computed from genuine NCI SMILES; they are not laboratory measurements.',
        fontsize=9,color='#526676')
    fig.text(.04,.013,'No newly generated compounds, target activities, synthesis, potency or safety are represented.',
        fontsize=8.6,color='#526676')
    fig.subplots_adjust(left=.04,right=.985,top=.875,bottom=.075,hspace=.31,wspace=.10)
    for ext in ('pdf','png','svg'):
        fig.savefig(out/f'original_nci_chemical_structures.{ext}',facecolor='white',dpi=190)
    plt.close(fig)
    report={'source_dataset':'RDKit bundled NCI first_5K.smi',
       'original_file_sha256':source_sha,
       'genuine_first_filtered_unique_molecules':600,
       'splits':{'training':len(train),'validation':len(val),'test':len(test)},
       'observed_source_structures_not_generations':True,
       'n_selected_original_molecules':len(records),'records':records,
       'no_claim':'Not generated molecules, not pharmacology or clinical validity'}
    (out/'original_nci_molecules_provenance.json').write_text(
        json.dumps(report,indent=2)+'\n')
    print(json.dumps({'source_sha256':source_sha,'source':600,
        'figure_records':12,'split_counts':report['splits']},indent=2))

if __name__=='__main__':main()
