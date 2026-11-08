"""Actual matched architecture comparison on original CrowdHuman annotations.

All detectors run on identical original photographs, COCO pretrained frozen
torchvision models, no local fine-tuning. Distinguishes bounding-box occlusion
proxy from visible pixel fraction. Local IoU50 recall only; NOT official MR/JI/AP.
"""
import argparse,csv,hashlib,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2
import torch
from torchvision.models.detection import (
 fasterrcnn_resnet50_fpn,FasterRCNN_ResNet50_FPN_Weights,
 fasterrcnn_mobilenet_v3_large_320_fpn,FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
)
HERE=Path(__file__).resolve()
CASE=HERE.parents[2]/"case_studies"/"occluded_pedestrian_detection"
sys.path.insert(0,str(CASE))
from evaluate_crowdhuman_frcnn import annotations,assign_recall
STRATA=("low","medium","heavy")
def bin_occlusion(v):
    # Stable binning at exact published decimal thresholds (0.20 and 0.50).
    # Avoid binary floating-point artifacts, e.g. 1 - 0.8 < 0.2.
    v = round(float(v), 12)
    return "low" if v < .2 else "medium" if v < .5 else "heavy"

def read_original(images,odgt,n):
    records=[json.loads(line) for line in odgt.read_text().splitlines() if line.strip()]
    if len(records)!=n: raise ValueError(f"Expected exactly {n} original chosen records, got {len(records)}")
    output=[]
    for entry in records:
        path=images/(entry["ID"]+".jpg")
        img=cv2.imread(str(path))
        if img is None:raise FileNotFoundError(path)
        gt=annotations(entry)
        output.append((entry["ID"],img,gt,hashlib.sha256(path.read_bytes()).hexdigest()))
    if len({r[0] for r in output})!=n:raise ValueError("Duplicate original frames")
    return output

def evaluate(model,weights,frames,conf):
    model=model.eval().cpu()
    person=[]
    for i,(name,image,gt,digest) in enumerate(frames,1):
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB).copy()
        inp=weights.transforms()(torch.from_numpy(rgb).permute(2,0,1).float()/255)
        with torch.inference_mode(): pred=model([inp])[0]
        bb=pred["boxes"].cpu().numpy()
        sc=pred["scores"].cpu().numpy()
        cls=pred["labels"].cpu().numpy()
        preds=[(box,float(score)) for box,score,label in zip(bb,sc,cls)
                if int(label)==1 and float(score)>=conf]
        hits=assign_recall(preds,gt)
        for j,((box,occ),hit) in enumerate(zip(gt,hits)):
            person.append({"image":name,"instance":j,"occlusion_bin":bin_occlusion(occ),
                          "occlusion_area_proxy":float(occ),"matched_IoU50":int(hit),
                          "image_sha256":digest})
        print(f"REAL {i}/{len(frames)} {name} people={len(gt)} hits={sum(hits)}",flush=True)
    return person

def summary(person):
    result={}
    for group in ("overall",)+STRATA:
        xs=[p for p in person if group=="overall" or p["occlusion_bin"]==group]
        result[group]={"matched":sum(p["matched_IoU50"] for p in xs),"count":len(xs)}
        result[group]["recall"]=result[group]["matched"]/max(1,len(xs))
    return result

def paired_bootstrap(data,keys,draws=5000,seed=20261108):
    rng=np.random.default_rng(seed)
    idxs=sorted(keys)
    out={k:[] for k in ("overall",)+STRATA}
    for _ in range(draws):
        sampled=rng.choice(idxs,size=len(idxs),replace=True)
        a=[row for name in sampled for row in data["MobileNetV3-FPN"][name]]
        b=[row for name in sampled for row in data["ResNet50-FPN"][name]]
        for group in out:
            aa=[r["matched_IoU50"] for r in a if group=="overall" or r["occlusion_bin"]==group]
            bb=[r["matched_IoU50"] for r in b if group=="overall" or r["occlusion_bin"]==group]
            if len(aa)!=len(bb) or not aa: raise RuntimeError("Lost real annotations in paired bootstrap")
            out[group].append((sum(bb)-sum(aa))/len(aa))
    return {k:list(map(float,np.quantile(v,[.025,.975]))) for k,v in out.items()}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--images",type=Path,required=True)
    p.add_argument("--annotations",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--num-images",type=int,default=48)
    p.add_argument("--score",type=float,default=.30)
    a=p.parse_args()
    if not(1<=a.num_images<=200 and 0<=a.score<1):raise ValueError("invalid parameters")
    torch.set_num_threads(4)
    a.out.mkdir(parents=True,exist_ok=True)
    frames=read_original(a.images,a.annotations,a.num_images)
    models=(
      ("MobileNetV3-FPN",fasterrcnn_mobilenet_v3_large_320_fpn,
       FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT),
      ("ResNet50-FPN",fasterrcnn_resnet50_fpn,
       FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    )
    bymodel={}
    for name,construct,weights in models:
        print("MODEL",name,weights.url,flush=True)
        bymodel[name]=evaluate(construct(weights=weights),weights,frames,a.score)
    original_order=[(x["image"],x["instance"],x["occlusion_bin"]) for x in bymodel["MobileNetV3-FPN"]]
    assert original_order==[(x["image"],x["instance"],x["occlusion_bin"]) for x in bymodel["ResNet50-FPN"]]
    grouped={name:defaultdict(list) for name in bymodel}
    for name,records in bymodel.items():
        for row in records:grouped[name][row["image"]].append(row)
    computed={name:summary(rows) for name,rows in bymodel.items()}
    intervals=paired_bootstrap(grouped,[f[0] for f in frames])
    report={"experiment":"Actual original CrowdHuman paired occlusion detection",
            "scope":"Local full-body IoU50 recall, NOT official CrowdHuman AP/MR/JI",
            "models":[{"name":name,"weights":weights.url} for name,_,weights in models],
            "source":"Funtion/CrowdHuman original validation JPEG/ODGT",
            "images":len(frames),"reference_people":len(bymodel["MobileNetV3-FPN"]),
            "score_threshold":a.score,"iou_threshold":.5,
            "occlusion_proxy":"1 - visible bounding box area/full bounding box area",
            "metrics":computed,
            "paired_difference_resnet_minus_mobilenet_CI95":intervals,
            "paired_bootstrap":"5000 resamples of original photograph clusters, seed 20261108",
            "occlusion_threshold_policy":"Round proxy to 12 decimal places before applying low <0.20, medium [0.20,0.50), heavy >=0.50"}
    with (a.out/"per_person_paired.csv").open("w",newline="") as f:
        fields=["model","image","instance","occlusion_bin","occlusion_area_proxy","matched_IoU50","image_sha256"]
        wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader()
        for model,rows in bymodel.items():
            wr.writerows({"model":model,**r} for r in rows)
    (a.out/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.25),layout="constrained")
    ix=np.arange(3)
    for k,(name,color) in enumerate([("MobileNetV3-FPN","#2B798D"),("ResNet50-FPN","#BD7950")]):
        scores=[computed[name][c]["recall"] for c in STRATA]
        axes[0].bar(ix+(-.18 if k==0 else .18),scores,width=.34,label=name,color=color)
        for i,v in enumerate(scores):axes[0].text(ix[i]+(-.18 if k==0 else .18),v+.015,f"{v:.3f}",ha="center",fontsize=8)
    axes[0].set(xticks=ix,xticklabels=STRATA,ylabel="Local person recall at IoU ≥ 0.50",ylim=(0,1.12))
    axes[0].legend(frameon=False,fontsize=8)
    for k,(name,color) in enumerate([("MobileNetV3-FPN","#2B798D"),("ResNet50-FPN","#BD7950")]):
        values=[sum(v["matched_IoU50"] for v in grouped[name][key])/max(1,len(grouped[name][key])) for key in sorted(grouped[name])]
        axes[1].scatter(range(len(values)),values,s=17,alpha=.64,color=color,label=name)
    axes[1].set(xlabel="Original validation-image index (same 48 images)",ylabel="Per-image person recall",ylim=(-.07,1.07))
    axes[1].legend(frameon=False,fontsize=8)
    for ax in axes:
        ax.grid(axis="y",alpha=.15)
        ax.set_axisbelow(True)
        ax.spines[["top","right"]].set_visible(False)
    for ext in ("pdf","png","svg"):
        fig.savefig(a.out/f"crowdhuman_actual_paired_occlusion.{ext}",dpi=210,bbox_inches="tight")
    print("FINISHED_MEASURED_RESULTS",json.dumps(report),flush=True)
if __name__=="__main__":main()
