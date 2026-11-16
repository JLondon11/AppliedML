"""Render two print-width original-tissue VAE-vs-oracle atlas plates.

Requires original 217 source-measured records and an actually trained VAE
checkpoint. No simulated data; prints real model outputs, reference photographs
and spatial errors at genuine 6.5-inch insertion width.
"""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np,torch
from skimage import data
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from audit_true_tissue_vae_vs_nearest import VAE,patches

SOURCE_SHA="c5b3ef509a92f16d4c29be8cf0300fe75d53e13a3ce650159db932caea8dcc1b"
PICKS=[125,9,156,40,50,96,159,35]

def render(checkpoint,records,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    photo=data.immunohistochemistry()
    if hashlib.sha256(photo.tobytes()).hexdigest()!=SOURCE_SHA:
        raise ValueError("Original source photo identity differs")
    train,_=patches(photo,0,320)
    test,_=patches(photo,384,512)
    if (len(train),len(test))!=(589,217):
        raise ValueError("Incorrect original training/test source")
    with Path(records).open(newline="",encoding="utf8") as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=217 or any(int(r["test_index"])!=i for i,r in enumerate(rows)):
        raise ValueError("Missing original 217 paired observations")
    checkpoint=Path(checkpoint)
    checkpoint_data=torch.load(checkpoint,map_location="cpu",weights_only=True)
    weights=checkpoint_data["model"] if "model" in checkpoint_data else checkpoint_data["weights"]
    model=VAE();model.load_state_dict(weights);model.eval()
    with torch.no_grad():
        selected=torch.from_numpy(test[PICKS])
        predictions=model.decode(model.encode(selected)[0]).numpy()
    channel_last=lambda x:np.moveaxis(x,0,-1)
    files=[]
    for group,indices in enumerate((PICKS[:4],PICKS[4:])):
        fig,axes=plt.subplots(4,4,figsize=(6.5,7.2),facecolor="white")
        fig.suptitle("Tissue VAE · measured reconstruction errors",
            x=.045,y=.977,ha="left",fontsize=12.5,fontweight="bold",color="#193f58")
        fig.text(.045,.932,"Panel "+str(group+1)+"/2: "+
            ("VAE-favorable observed cases" if group==0 else "Oracle-favorable observed cases"),
            fontsize=10.3,color="#176f7b" if group==0 else "#9c5e45",weight="bold")
        fig.text(.045,.901,"One original tissue photograph · 217 correlated held-out source patches",
            fontsize=8.7,color="#486477")
        for column,index in enumerate(indices):
            item=rows[index]
            neighbor_index=int(item["nearest_train_index"])
            actual=channel_last(test[index]);recon=channel_last(predictions[group*4+column])
            neighbor=channel_last(train[neighbor_index])
            residual=np.abs(actual-recon).mean(2)
            for row,image in enumerate((actual,recon,neighbor,residual)):
                ax=axes[row,column]
                ax.imshow(image,cmap="magma" if row==3 else None,vmin=0,
                    vmax=.4 if row==3 else 1,interpolation="nearest")
                ax.set_xticks([]);ax.set_yticks([])
            delta=float(item["nearest_minus_vae_mse"])
            axes[0,column].set_title(f"Patch {index}\nΔ {delta:+.4f}",fontsize=8.6)
        for row,label in enumerate(
            ("Real test image","Executed VAE output","Oracle-nearest real patch","Absolute RGB error")):
            axes[row,0].set_ylabel(label,fontsize=8.5,labelpad=8)
        fig.text(.045,.066,
            "Target-aware oracle-nearest uses the true held-out image; it is not a deployable predictor.",
            fontsize=8.4,color="#3e5d71")
        fig.text(.045,.038,
            "Extremes selected, not random. Error color scale 0–0.4; no independent patient data.",
            fontsize=8.1,color="#3e5d71")
        fig.subplots_adjust(left=.19,right=.985,top=.854,bottom=.105,wspace=.18,hspace=.29)
        name=f"actual_tissue_oracle_paired_print_page_{group+1}"
        for ext in ("pdf","svg","png"):
            fig.savefig(out/f"{name}.{ext}",dpi=220,facecolor="white")
        plt.close(fig)
        files.append(str(out/f"{name}.pdf"))
    manifest={
        "original_source_photographs":1,"original_test_patches":217,
        "actual_observed_patch_ids":PICKS,
        "source_pixel_sha256":SOURCE_SHA,
        "executed_checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "figure_width_inches":6.5,
        "baseline_warning":"Oracle nearest library photograph is selected with original held-out target pixels, not a deployable matched-input predictor",
        "staged_pdf_files":files,
        "manuscript_insertion_approved":False}
    (out/"print_width_atlas_provenance.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True,type=Path)
    p.add_argument("--records",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    render(a.checkpoint,a.records,a.output)
