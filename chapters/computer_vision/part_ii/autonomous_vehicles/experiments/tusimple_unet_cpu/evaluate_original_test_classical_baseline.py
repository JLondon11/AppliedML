"""Original TuSimple test-set fixed color-threshold baseline vs executed U-Net.

The U-Net inputs and outputs are inherited UNCHANGED from original GitHub run
42432540622, not re-synthesized or inferred. The classical baseline is measured
on the 32 identical ORIGINAL images, with equal 4-pixel GT rasterization.
This is a local pixel-mask diagnostic, NOT official TuSimple lane-point score.
"""
import argparse,csv,hashlib,json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO="dhbloo/TuSimple"

def original_ground_truth(record,shape):
    height,width=shape
    mask=np.zeros((height,width),np.uint8)
    for lane in record["lanes"]:
        if len(lane)!=len(record["h_samples"]):
            raise RuntimeError("Corrupted original lane annotations")
        pts=[(int(x),int(y)) for x,y in zip(lane,record["h_samples"])
             if 0 <= int(x)<width and 0 <= int(y)<height]
        if len(pts)>1: cv2.polylines(mask,[np.array(pts,np.int32)],False,1,thickness=4)
    return mask.astype(bool)

def fixed_color_lane_mask(frame):
    """Predeclared white/yellow HSV lane-color heuristic, no fitting or tuning."""
    h,w=frame.shape[:2]
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    white=cv2.inRange(hsv,(0,0,175),(179,72,255))
    yellow=cv2.inRange(hsv,(12,65,85),(42,255,255))
    pixels=cv2.bitwise_or(white,yellow)
    # Fixed trapezoidal drivable-road region, rather than unrestricted sky/horizon.
    roi=np.zeros((h,w),np.uint8)
    shape=np.array([[(int(.07*w),h-1),(int(.93*w),h-1),
                     (int(.63*w),int(.52*h)),(int(.37*w),int(.52*h))]],np.int32)
    cv2.fillPoly(roi,shape,255)
    return ((pixels>0)&(roi>0))

def measurements(p,g):
    tp=int(np.count_nonzero(p&g))
    fp=int(np.count_nonzero(p&~g))
    fn=int(np.count_nonzero(~p&g))
    return {"tp":tp,"fp":fp,"fn":fn,"iou":tp/max(1,tp+fp+fn),
            "dice":2*tp/max(1,2*tp+fp+fn)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--reference-csv",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seed",type=int,default=20261108)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    records={r["raw_file"]:r for r in [
        json.loads(t) for t in Path(hf_hub_download(REPO,"test_label.json",repo_type="dataset"))
        .read_text().splitlines() if t.strip()]}
    with a.reference_csv.open(newline="") as f:
        refs=list(csv.DictReader(f))
    if len(refs)!=32 or len(set(r["frame"] for r in refs))!=32:
        raise ValueError("Source pretrained-model evidence must have 32 distinct original images")
    outputs=[]
    for i,r in enumerate(refs,1):
        name=r["frame"]
        if name not in records:raise ValueError("Missing original test annotation "+name)
        filepath=Path(hf_hub_download(REPO,name,repo_type="dataset"))
        blob=filepath.read_bytes()
        sha=hashlib.sha256(blob).hexdigest()
        if sha!=r["image_sha256"]:
            raise ValueError("Original photograph differs from frozen real U-Net experiment: "+name)
        img=cv2.imdecode(np.frombuffer(blob,np.uint8),cv2.IMREAD_COLOR)
        if img is None or img.shape[:2]!=(720,1280):raise ValueError("Bad original frame dimensions")
        gt=original_ground_truth(records[name],img.shape[:2])
        if int(gt.sum())!=int(r["gt_pixels"]):
            raise ValueError("Ground-truth rasterization differs from executed U-Net evaluation")
        baseline=fixed_color_lane_mask(img)
        local=measurements(baseline,gt)
        assert local["tp"]+local["fn"]==int(r["gt_pixels"])
        outputs.append({"frame":name,"gt_pixels":int(gt.sum()),"image_sha256":sha,
                        **{f"baseline_{k}":v for k,v in local.items()},
                        **{f"unet_{k}":float(r[k]) if k in ("iou","dice") else int(r[k])
                           for k in ("tp","fp","fn","iou","dice")}})
        print(f"real original test image {i}/32  U-Net={r['iou']} baseline={local['iou']:.6f}",flush=True)

    with (a.out/"per_image_comparison.csv").open("w",newline="") as f:
        fields=list(outputs[0])
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader();writer.writerows(outputs)
    metrics={}
    for label in ("baseline","unet"):
        counts=[sum(int(x[f"{label}_{k}"]) for x in outputs) for k in ("tp","fp","fn")]
        tp,fp,fn=counts
        metrics[label]={"micro_iou":tp/max(1,tp+fp+fn),
                        "micro_dice":2*tp/max(1,2*tp+fp+fn),
                        "macro_iou":float(np.mean([float(x[f"{label}_iou"]) for x in outputs])),
                        "micro_precision":tp/max(1,tp+fp),"micro_recall":tp/max(1,tp+fn)}
    assert abs(metrics["unet"]["micro_iou"]-0.17983022473122395)<1e-12
    assert abs(metrics["unet"]["micro_dice"]-0.30484085076255935)<1e-12
    ci=[]
    rng=np.random.default_rng(a.seed)
    for _ in range(5000):
        ind=rng.integers(0,len(outputs),len(outputs))
        x=[outputs[int(j)] for j in ind]
        delta=(sum(t["unet_tp"] for t in x)/max(1,sum(t["unet_tp"]+t["unet_fp"]+t["unet_fn"] for t in x)) -
               sum(t["baseline_tp"] for t in x)/max(1,sum(t["baseline_tp"]+t["baseline_fp"]+t["baseline_fn"] for t in x)))
        ci.append(delta)
    metrics["delta_unet_minus_baseline_micro_iou"]=metrics["unet"]["micro_iou"]-metrics["baseline"]["micro_iou"]
    metrics["delta_micro_iou_bootstrap_ci95"]=list(map(float,np.quantile(ci,[.025,.975])))
    metrics["evaluation"]={"original_dataset":REPO,"source_partition":"test_label.json",
            "image_count":len(outputs),"U_Net_recorded_run":42432540622,
            "paired_original_photo_sha256_verified":True,
            "reference_rasterization":"4 pixel polylines native resolution",
            "classic_baseline":"Predeclared HSV white/yellow + fixed trapezoid ROI; no training or tuning",
            "baseline_training_count":0,"unet_local_training_count":0,
            "checkpoint_training_membership_verified":False,
            "bootstrap":"5,000 original-image clustered replicates, seed 20261108",
            "scope":"Local mask IoU comparison; NOT TuSimple official accuracy; no independent model-training holdout proof"}
    (a.out/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    fig,ax=plt.subplots(1,2,figsize=(10.6,4.1),layout="constrained")
    xx=np.arange(32)
    ax[0].scatter(xx,[r["unet_iou"] for r in outputs],s=24,c="#287F95",label="Pretrained U-Net")
    ax[0].scatter(xx,[r["baseline_iou"] for r in outputs],s=24,c="#B97952",marker="D",label="Fixed HSV heuristic")
    ax[0].set(xlabel="Same original TuSimple test photograph (seeded order)",ylabel="4-pixel mask IoU",ylim=(0,.55))
    ax[0].legend(fontsize=8,frameon=False)
    ax[1].scatter([r["baseline_iou"] for r in outputs],[r["unet_iou"] for r in outputs],
                  c="#287F95",s=35,alpha=.8)
    ax[1].plot([0,.55],[0,.55],ls="--",color="#777",lw=1)
    ax[1].set(xlabel="Fixed HSV heuristic mask IoU",ylabel="Pretrained U-Net mask IoU",xlim=(0,.55),ylim=(0,.55))
    ax[1].set_aspect("equal","box")
    for a0 in ax:
        a0.grid(alpha=.12);a0.set_axisbelow(True)
        a0.spines[["top","right"]].set_visible(False)
    for ext in ("pdf","png","svg"):
        fig.savefig(a.out/f"tusimple_paired_original_test_real.{ext}",dpi=210,bbox_inches="tight")
    print("PAIRED_REAL_TUSIMPLE_RESULTS",json.dumps(metrics),flush=True)

if __name__=="__main__":main()
