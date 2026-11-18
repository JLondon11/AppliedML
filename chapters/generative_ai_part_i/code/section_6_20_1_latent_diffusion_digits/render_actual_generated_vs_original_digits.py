"""Publication-sized genuine-image-versus-executed-diffusion galleries.

The only inputs are trusted outputs from a successful ORIGINAL GitHub Actions
run of Chapter 6's native class-conditioned latent digit diffusion experiment:
2000 actually generated tensors, per-image classifier predictions and the
genuine original 270-example heldout split. No synthetic test observations are
substituted; diagram is genuine model output, not Stable Diffusion.
"""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
import torch
from sklearn.datasets import load_digits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOURCE_SHA="f6d9e39f37dc45d327f6db33428ee58970ccceabb2535a5c179de35886b70443"
ARMS=("conditional","unconditional")
def render(artifact,out):
    artifact=Path(artifact)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((artifact/"manifest.json").read_text())
    original=load_digits()
    Y=original.target.astype("int64")
    source_hash=hashlib.sha256(original.data.tobytes()+Y.tobytes()).hexdigest()
    if source_hash!=SOURCE_SHA or manifest["source_sha256"]!=SOURCE_SHA:
        raise ValueError("Original real-digit source mismatch")
    # These are our own executed GitHub Actions PyTorch files; never unpickle
    # untrusted externally supplied checkpoints or arbitrary files.
    split=torch.load(artifact/"split_indices.pt",map_location="cpu",weights_only=False)
    train=set(map(int,split["train"]));val=set(map(int,split["validation"]))
    test=np.asarray(split["test"],dtype=int)
    if (len(train),len(val),len(test))!=(1257,270,270):
        raise ValueError("Genuine original scan partition size mismatch")
    if train.intersection(test) or val.intersection(test) or train.intersection(val):
        raise ValueError("Leakage across original source-scan partitions")
    data={}
    for arm in ARMS:
        d=torch.load(artifact/f"generated_{arm}.pt",map_location="cpu",weights_only=False)
        matrix=d["generated"]
        if hasattr(matrix,"detach"):matrix=matrix.detach().cpu().numpy()
        matrix=np.asarray(matrix,dtype=float)
        prompts=np.asarray(d["prompts"],dtype=int)
        if matrix.shape!=(1000,64) or prompts.shape!=(1000,):
            raise ValueError("Missing original generated model tensor rows")
        if not np.array_equal(prompts,np.repeat(np.arange(10),100)):
            raise ValueError("Saved source model prompts not intact")
        if not np.isfinite(matrix).all():
            raise ValueError("Nonfinite output pixel intensities")
        data[arm]=matrix
    with (artifact/"predictions.csv").open(newline="",encoding="utf8") as f:
        records=list(csv.DictReader(f))
    if len(records)!=2000:raise ValueError("Original model output probability file incomplete")
    by={}
    for record in records:
        arm=record["arm"];i=int(record["index"])
        if arm not in ARMS or not 0<=i<1000 or (arm,i) in by:
            raise ValueError("Missing/duplicate original model-output prediction key")
        true_class=i//100
        if int(record["prompt_label"])!=true_class:
            raise ValueError("Original row prompt is not aligned with generated pixel tensor")
        p=np.array([float(record[f"p_{k}"]) for k in range(10)])
        if abs(p.sum()-1)>1e-5:
            raise ValueError("Original prediction probabilities do not sum to 1")
        pred=int(record["classifier_prediction"])
        if np.argmax(p)!=pred:
            raise ValueError("Saved original classifier output inconsistent with probabilities")
        by[(arm,i)]=dict(actual_prediction=pred,requested_digit=true_class,
                         source_probability=float(p[pred]),
                         requested_digit_probability=float(p[true_class]))
    if len(by)!=2000:raise ValueError("Original digit prediction output count mismatch")
    totals={}
    for arm in ARMS:
        totals[arm]=sum(by[(arm,i)]["actual_prediction"]==i//100 for i in range(1000))
    if any(abs(totals[a]/1000-manifest[
            "class_conditional_consistency" if a=="conditional" else "unconditional_consistency"])>1e-12
           for a in ARMS):
        raise ValueError("Actual sample outcomes not reconciled to original run manifest")
    # First real heldout photograph for each digit class; first actual
    # conditional and unconditional sample from each requested class.
    # Fixed positional selection; no cherry-picking on visual quality or predictions.
    source_ids={digit:next(int(i) for i in test if Y[i]==digit) for digit in range(10)}
    entries=[]
    colors={"conditional":"#196f79","unconditional":"#b26c4c"}
    for page_no,classes in enumerate((range(0,5),range(5,10)),start=1):
        fig,axes=plt.subplots(3,5,figsize=(6.5,6.8),facecolor="white")
        for column,digit in enumerate(classes):
            selected_index=100*digit
            images=[
                (original.images[source_ids[digit]]/16., "original",None),
                (data["conditional"][selected_index].reshape(8,8),
                 "conditional",by[("conditional",selected_index)]),
                (data["unconditional"][selected_index].reshape(8,8),
                 "unconditional",by[("unconditional",selected_index)])]
            for row,(pixels,arm,result) in enumerate(images):
                ax=axes[row,column]
                ax.imshow(pixels,cmap="gray",vmin=0,vmax=1,
                          interpolation="nearest")
                ax.set_xticks([]);ax.set_yticks([])
                if row==0:ax.set_title(f"Digit {digit}",fontsize=10.4,
                                       color="#1b425a",weight="bold")
                if result is not None:
                    correct=result["actual_prediction"]==digit
                    ax.text(.04,.035,f"pred {result['actual_prediction']}" +
                            (" ✓" if correct else " ×"),
                            transform=ax.transAxes,va="bottom",ha="left",fontsize=8.1,
                            color="white",weight="bold",
                            bbox=dict(facecolor=colors[arm],alpha=.92,
                                      edgecolor="none",pad=1.5))
                entry=dict(page=page_no,source_class=digit,kind=arm,
                           source_position=source_ids[digit] if row==0 else selected_index,
                           pixel_sha256=hashlib.sha256(
                               np.asarray(pixels,dtype="<f4").tobytes()).hexdigest())
                if result is not None:entry.update(result)
                entries.append(entry)
        for row,label in enumerate(("Held-out real\nsource scan",
                                    "Actual conditional\nmodel output",
                                    "Actual unconditional\nmodel output")):
            axes[row,0].set_ylabel(label,fontsize=8.3,labelpad=10)
        fig.text(.055,.964,"Original scans ↔ executed latent diffusion",
                 fontsize=14.5,color="#183c54",weight="bold")
        fig.text(.055,.924,f"Digits {classes.start}–{classes.stop-1} · genuine test scans versus two saved generator arms",
                 fontsize=8.9,color="#536a7a")
        fig.text(.055,.075,
                 f"Entire executed run: conditional {totals['conditional']}/1,000; unconditional {totals['unconditional']}/1,000 classifier matches.",
                 fontsize=8.6,color="#24495e")
        fig.text(.055,.049,"Each column uses the first source test scan and first generated output for its class.",
                 fontsize=8.1,color="#536a7a")
        fig.text(.055,.025,"8×8 original pixels; no natural-language prompting, clinical data or independent seed replication.",
                 fontsize=8.1,color="#536a7a")
        fig.subplots_adjust(left=.18,right=.985,top=.87,bottom=.14,hspace=.22,wspace=.12)
        for ext in ("pdf","png","svg"):
            fig.savefig(out/f"actual_original_vs_diffusion_digits_{page_no}.{ext}",
                        facecolor="white",dpi=255)
        plt.close(fig)
    result={"original_source_sha256":SOURCE_SHA,
            "original_train_scans":1257,"original_val_scans":270,
            "original_test_scans":270,
            "actual_generated_rows":2000,"actual_generated_tensors":{a:1000 for a in ARMS},
            "observed_classifier_matches":totals,"selected_samples":entries,
            "model_run_provenance":manifest,
            "sample_selection":"First held-out real source photo per class; first original generated tensor per class in each model arm; no score-based or visual selection",
            "limitations":"Original low-resolution 8x8 handwritten digits; classifier agreement is indirect and no independent training-seed performance estimated.",
            "manuscript_production_approved":False}
    (out/"actual_digit_image_comparison_provenance.json").write_text(
        json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("selected_samples","model_run_provenance")},indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--artifact",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args();render(a.artifact,a.output)
