"""Print-width, genuine-data image plates for Chapter 6.

The tissue output plates require an executed checkpoint. Other plates depict
original inputs, NOT generated or experimentally assayed results.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np,torch
from rdkit import Chem,RDConfig
from rdkit.Chem import Draw,rdMolDescriptors
from skimage import data
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

TISSUE="c5b3ef509a92f16d4c29be8cf0300fe75d53e13a3ce650159db932caea8dcc1b"
NCI="91e71c015f14939837f2943dcc904f7c87e5a3a0124d82b05c28ad2f23004def"
INK="#183d54";MUTED="#536b79"
def save(fig,out,name):
    for ext in ("pdf","png","svg"):fig.savefig(out/f"{name}.{ext}",dpi=240,facecolor="white")
    plt.close(fig)
def crops(photo,lo,hi):
    result=[]
    for row in range(lo,hi-31,16):
        for col in range(0,512-31,16):
            result.append(np.moveaxis(photo[row:row+32,col:col+32],-1,0))
    return np.stack(result).astype("float32")/255
def source_figures(out):
    photo=data.immunohistochemistry()
    if hashlib.sha256(photo.tobytes()).hexdigest()!=TISSUE:
        raise ValueError("Original microscopy identity differs")
    fig=plt.figure(figsize=(6.5,7.4),facecolor="white")
    gs=fig.add_gridspec(3,3,left=.065,right=.98,top=.87,bottom=.115,wspace=.12,hspace=.16,height_ratios=[2.25,1,1])
    ax=fig.add_subplot(gs[0,:]);ax.imshow(photo);ax.axis("off")
    for a,b,color,label in ((0,320,"#147f78","TRAIN"),(320,384,"#b88735","VALIDATION"),(384,512,"#a45d6a","TEST")):
        ax.add_patch(Rectangle((-.5,a-.5),512,b-a,edgecolor=color,fill=False,lw=3))
        ax.text(7,a+14,label,fontsize=9,weight="bold",color="white",
            bbox=dict(facecolor=color,edgecolor="none",pad=2))
    for i,(row,col) in enumerate([(32,32),(144,224),(256,384),(336,48),(400,224),(464,416)]):
        a=fig.add_subplot(gs[1+i//3,i%3]);a.imshow(photo[row:row+32,col:col+32],interpolation="nearest")
        a.axis("off");a.set_title(f"Row {row}, col {col}",fontsize=8.5,color=INK)
    fig.text(.065,.949,"Actual tissue data and spatial splits",fontsize=15,weight="bold",color=INK)
    fig.text(.065,.930,"One 512 × 512 original micrograph; real 32 × 32 input crops",fontsize=9,color=MUTED)
    fig.text(.065,.075,"Original pixels, NOT VAE predictions. Train 0–319; validation 320–383; test 384–511.",fontsize=8.2,color=MUTED)
    fig.text(.065,.044,"Spatial patches overlap within regions; the specimen photograph is not a patient cohort.",fontsize=8.2,color=MUTED)
    save(fig,out,"tissue_source_compact")
    digits=load_digits()
    fig,axes=plt.subplots(4,5,figsize=(6.5,6.6),facecolor="white")
    for digit in range(10):
        ids=np.flatnonzero(digits.target==digit)
        for variant,idx in enumerate((int(ids[0]),int(ids[len(ids)//2]))):
            row=(digit//5)*2+variant;col=digit%5
            a=axes[row,col];a.imshow(digits.images[idx],cmap="gray",vmin=0,vmax=16,interpolation="nearest")
            a.set_xticks([]);a.set_yticks([])
            if variant==0:a.set_title(f"Class {digit}",fontsize=9.2,weight="bold",color=INK)
    fig.text(.055,.946,"Authentic scanned handwriting",fontsize=15,weight="bold",color=INK)
    fig.text(.055,.928,"Two original 8 × 8 scans per digit class — not generated images",fontsize=9,color=MUTED)
    fig.text(.055,.047,"Original sklearn digit scans shown at nearest-neighbor pixel scale; intensities 0–16.",fontsize=8.2,color=MUTED)
    fig.subplots_adjust(left=.055,right=.98,bottom=.105,top=.85,wspace=.13,hspace=.22)
    save(fig,out,"digits_source_compact")
    path=Path(RDConfig.RDDataDir)/"NCI"/"first_5K.smi"
    if hashlib.sha256(path.read_bytes()).hexdigest()!=NCI:raise ValueError("Original NCI SHA256 changed")
    original=[];seen=set()
    for line in path.read_text().splitlines():
        parts=line.split()
        if len(parts)<2:continue
        mol=Chem.MolFromSmiles(parts[0])
        if mol is None:continue
        s=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=False)
        if len(s)>75 or s in seen:continue
        original.append((mol,parts[1]));seen.add(s)
        if len(original)==600:break
    if len(original)!=600:raise ValueError("Original NCI source subset incomplete")
    ids=np.arange(600);train,test=train_test_split(ids,test_size=.2,random_state=20261109)
    train,val=train_test_split(train,test_size=.25,random_state=20261109)
    selected=[]
    for label,indices in (("Train",train),("Validation",val),("Test",test)):
        seq=sorted(map(int,indices))
        selected.extend((label,seq[k]) for k in (1,len(seq)//2))
    fig,axes=plt.subplots(3,2,figsize=(6.5,8.0),facecolor="white")
    for a,(label,i) in zip(axes.flat,selected):
        mol,nci=original[i];a.imshow(Draw.MolToImage(mol,size=(400,265)));a.axis("off")
        a.set_title(f"{label} · original NCI {nci}  |  {rdMolDescriptors.CalcMolFormula(mol)}",
                    color=INK,fontsize=8.5,loc="left")
    fig.text(.055,.953,"Original NCI molecular structures",color=INK,fontsize=14.5,weight="bold")
    fig.text(.055,.940,"Six authentic source compounds, not VAE-generated molecules",color=MUTED,fontsize=9)
    fig.text(.055,.047,"Train/validation/test: 360/120/120 distinct original source structures.",color=MUTED,fontsize=8.2)
    fig.text(.055,.021,"Bond diagrams and formulas are computed from NCI SMILES; no new chemical assays.",color=MUTED,fontsize=8.2)
    fig.subplots_adjust(left=.055,right=.99,top=.88,bottom=.085,hspace=.27,wspace=.13)
    save(fig,out,"nci_source_compact")
    return photo
def actual_model_figures(out,photo,checkpoint):
    # Reuse the exact model class already committed under the same study.
    import sys
    original_model_dir=Path(__file__).resolve().parents[0].parent/"code"/"section_6_medical_imaging_histology_vae"
    if not (original_model_dir/"audit_true_tissue_vae_vs_nearest.py").is_file():
        raise FileNotFoundError("The committed original tissue VAE model analysis is missing")
    sys.path.insert(0,str(original_model_dir))
    from audit_true_tissue_vae_vs_nearest import VAE
    x=crops(photo,384,512)
    if len(x)!=217:raise ValueError("Original test patch count differs")
    record=torch.load(checkpoint,map_location="cpu",weights_only=True)
    model=VAE();model.load_state_dict(record.get("model",record.get("weights")));model.eval()
    chosen=[2,31,59,93,125,158,186,213]
    with torch.no_grad():
        xx=torch.from_numpy(x[chosen]);pred=model.decode(model.encode(xx)[0]).numpy()
        torch.manual_seed(20261109+111);prior=model.decode(torch.randn(8,24)).numpy()
    rgb=lambda a:np.moveaxis(a,0,-1)
    for part in range(2):
        fig,axes=plt.subplots(4,4,figsize=(6.5,7.25),facecolor="white")
        for col in range(4):
            i=part*4+col;a=rgb(x[chosen[i]]);b=rgb(pred[i]);c=rgb(prior[i]);err=np.abs(a-b).mean(2)
            for row,img in enumerate((a,b,err,c)):
                ax=axes[row,col];ax.imshow(img,cmap="magma" if row==2 else None,
                  vmin=0,vmax=.4 if row==2 else 1,interpolation="nearest")
                ax.set_xticks([]);ax.set_yticks([])
            axes[0,col].set_title(f"Original patch {chosen[i]}",fontsize=8.2,color=INK)
        for row,label in enumerate(("Original test","VAE recon","Absolute RGB error","Prior generation")):
            axes[row,0].set_ylabel(label,fontsize=8.2,labelpad=8)
        fig.text(.055,.967,"Real tissue versus actual VAE outputs",color=INK,fontsize=14.5,weight="bold")
        fig.text(.055,.929,f"Page {part+1}/2 · true held-out reconstructions and prior samples",fontsize=8.9,color=MUTED)
        fig.text(.055,.059,"One source micrograph; reconstructions measured on spatial held-out patches only.",fontsize=8.2,color=MUTED)
        fig.text(.055,.031,"Prior samples are artificial neural outputs, not independent tissue photographs.",fontsize=8.2,color=MUTED)
        fig.subplots_adjust(left=.19,right=.985,top=.855,bottom=.102,wspace=.17,hspace=.28)
        save(fig,out,f"tissue_vae_output_compact_{part+1}")
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    photo=source_figures(args.output)
    actual_model_figures(args.output,photo,args.checkpoint)
    (args.output/"print_compact_provenance.json").write_text(json.dumps({
        "source":"Authentic datasets and executed VAE checkpoint",
        "tissue_original_source_sha256":hashlib.sha256(photo.tobytes()).hexdigest(),
        "checkpoint_sha256":hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "staged_figure_count":5,"physical_candidate_width_inches":6.5,
        "not_full_manuscript_production_approval":True},indent=2)+"\n")
if __name__=="__main__":main()
