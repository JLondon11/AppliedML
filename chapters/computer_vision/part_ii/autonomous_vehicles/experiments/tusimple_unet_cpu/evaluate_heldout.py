"""Real held-out TuSimple TEST annotations and exact original photo pairs.

Runs pretrained Keras U-Net on a seeded subset of public original test frames.
No synthetic data. No claim of official TuSimple lane-point accuracy or
independence from a checkpoint's unknown training membership.
"""
import os
os.environ["KERAS_BACKEND"]="torch"
os.environ["OMP_NUM_THREADS"]="4"
import argparse,csv,hashlib,json,random
from pathlib import Path
import numpy as np,cv2,keras,torch
from huggingface_hub import HfApi,hf_hub_download
from evaluate import gt_mask,scores

REPO="dhbloo/TuSimple"
WEIGHTS="Tilak1812/lane-detection-unet-tusimple"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path("tusimple_heldout"))
    p.add_argument("--images",type=int,default=32)
    p.add_argument("--seed",type=int,default=20261107)
    p.add_argument("--threshold",type=float,default=.5)
    a=p.parse_args()
    if not (1<=a.images<=233 and 0<a.threshold<1):
        p.error("Invalid evaluation inputs")
    torch.set_num_threads(4)
    a.out.mkdir(parents=True,exist_ok=True)
    records=[json.loads(s) for s in Path(hf_hub_download(REPO,"test_label.json",repo_type="dataset")).read_text().splitlines() if s.strip()]
    original_paths={entry.path for entry in HfApi().list_repo_tree(REPO,repo_type="dataset",recursive=True)}
    eligible=sorted([r for r in records if r["raw_file"] in original_paths],key=lambda r:r["raw_file"])
    if len(eligible)<a.images:
        raise RuntimeError("No sufficiently large original test image/label pairing")
    chosen=random.Random(a.seed).sample(eligible,a.images)
    checkpoint=Path(hf_hub_download(WEIGHTS,"best_model.keras"))
    model=keras.saving.load_model(checkpoint,compile=False,safe_mode=True)
    h_in,w_in=map(int,model.inputs[0].shape[1:3])
    if len(model.outputs)!=1 or model.outputs[0].shape[-1]!=1:
        raise ValueError("Wrong pretrained segmentation model")
    rows=[]
    for i,r in enumerate(chosen):
        original=Path(hf_hub_download(REPO,r["raw_file"],repo_type="dataset"))
        image=cv2.imread(str(original))
        if image is None or image.shape[:2]!=(720,1280):
            raise RuntimeError("Original test image unavailable or unexpected resolution")
        gt=gt_mask(r,image.shape[:2],thickness=4)
        if gt.sum()<1:
            raise ValueError("Original test annotation missing lanes")
        scaled=cv2.cvtColor(cv2.resize(image,(w_in,h_in)),cv2.COLOR_BGR2RGB)
        pred=np.asarray(model.predict((scaled.astype(np.float32)/255)[None],verbose=0))
        if pred.shape!=(1,h_in,w_in,1):
            raise ValueError("Unknown pretrained model output shape")
        heat=pred[0,:,:,0]
        if not np.isfinite(heat).all():raise ValueError("Invalid predictions")
        if heat.min()<0 or heat.max()>1:
            heat=1/(1+np.exp(-np.clip(heat,-40,40)))
        predicted=cv2.resize(heat,(1280,720),interpolation=cv2.INTER_LINEAR)>=a.threshold
        tp,fp,fn,precision,recall,iou,dice=scores(predicted,gt)
        rows.append({"frame":r["raw_file"],"gt_pixels":int(gt.sum()),
                     "predicted_pixels":int(predicted.sum()),"tp":tp,"fp":fp,
                     "fn":fn,"precision":precision,"recall":recall,"iou":iou,
                     "dice":dice,"image_sha256":hashlib.sha256(original.read_bytes()).hexdigest()})
        print(f"FRAME {i+1}/{len(chosen)} IoU={iou:.5f} Dice={dice:.5f}",flush=True)
    with (a.out/"per_frame.csv").open("w",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=rows[0].keys())
        writer.writeheader();writer.writerows(rows)
    tp=sum(x["tp"] for x in rows);fp=sum(x["fp"] for x in rows);fn=sum(x["fn"] for x in rows)
    summary={
        "status":"pretrained_model_executed_original_TuSimple_test_photographs",
        "dataset":"Original TuSimple test_label.json and matched original test frames",
        "source_mirror":REPO,"source_test_annotation_count":len(records),
        "accessible_image_annotation_pairs":len(eligible),
        "selected_test_images":len(chosen),
        "training_data_in_evaluation":False,
        "checkpoint_training_membership_verified":False,
        "evaluation_protocol":"4-pixel ground-truth lane rasterization vs 0.5 U-Net mask; not official lane-point",
        "model":WEIGHTS,"seed":a.seed,"device":"cpu",
        "micro_iou":tp/max(1,tp+fp+fn),
        "micro_dice":2*tp/max(1,2*tp+fp+fn),
        "micro_precision":tp/max(1,tp+fp),
        "micro_recall":tp/max(1,tp+fn),
        "macro_iou":float(np.mean([x["iou"] for x in rows])),
        "checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    }
    (a.out/"metrics.json").write_text(json.dumps(summary,indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.2),layout="constrained")
    ix=np.arange(len(rows))
    ax.scatter(ix,[x["iou"] for x in rows],s=22,label="Mask IoU")
    ax.scatter(ix,[x["dice"] for x in rows],s=20,label="Mask Dice",marker="D")
    ax.axhline(summary["macro_iou"],ls="--",lw=1.2,label="Mean frame IoU")
    ax.set(xlabel="Seed-selected test photograph",ylabel="Lane rasterization overlap",
           ylim=(0,1.04))
    ax.legend(frameon=False);ax.grid(alpha=.16)
    for ext in ("pdf","png","svg"):
        fig.savefig(a.out/f"tusimple_original_test_measured.{ext}",dpi=220,bbox_inches="tight")
    print(json.dumps({k:v for k,v in summary.items() if not k.endswith("sha256")},indent=2),flush=True)
if __name__=="__main__":main()
