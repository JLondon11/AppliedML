"""Measured COCO-pretrained Faster R-CNN domain transfer, original Penn-Fudan vs CrowdHuman.

Identical trained weights, class, score threshold and IoU50 one-to-one matching.
Both sources are original photographs and original annotations. Separate subsets.
This is pedestrian-domain transfer, not autonomous-driving benchmark mAP.
"""
import argparse
import csv
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

PENN_URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"

def iou(a,b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1])
    bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/max(1e-12,aa+bb-inter)

def match_boxes(predictions,references,threshold=.5):
    matched=set();hits=[]
    for box,score in sorted(predictions,key=lambda x:-x[1]):
        ranked=sorted(enumerate(references),key=lambda ix:iou(box,ix[1]),reverse=True)
        for idx,ref in ranked:
            if idx not in matched and iou(box,ref)>=threshold:
                matched.add(idx);hits.append((idx,float(score)));break
    return matched

def penn_annotation_mask(mask):
    pix=np.array(Image.open(mask))
    if pix.ndim!=2:
        raise ValueError("Instance ID mask must be 2D")
    labels=np.unique(pix)
    result=[]
    for idx in labels[labels!=0]:
        ys,xs=np.where(pix==idx)
        if not len(xs): continue
        result.append([int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)])
    return result

def get_penn_zip(path):
    if not path.is_file():
        path.parent.mkdir(exist_ok=True,parents=True)
        with urllib.request.urlopen(PENN_URL,timeout=240) as source, path.open("wb") as target:
            while True:
                block=source.read(1<<20)
                if not block: break
                target.write(block)
    if path.stat().st_size<1_000_000:
        raise RuntimeError("Original PennFudanPed.zip not downloaded")
    return path

def read_penn(path,n):
    samples=[]
    with zipfile.ZipFile(path) as z:
        names=set(z.namelist())
        frames=sorted(x for x in names if x.startswith("PennFudanPed/PNGImages/") and x.endswith(".png"))
        if len(frames)<n:
            raise RuntimeError(f"Only {len(frames)} original Penn-Fudan frames")
        for filename in frames[:n]:
            mask_name="PennFudanPed/PedMasks/"+Path(filename).stem+"_mask.png"
            if mask_name not in names:
                raise RuntimeError(f"Original annotation missing: {mask_name}")
            image=cv2.imdecode(np.frombuffer(z.read(filename),np.uint8),cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError("Original Penn-Fudan image failed decoding")
            # Decode original PNG instance ID mask without changing labels.
            with z.open(mask_name) as handle:
                gt_pix=np.array(Image.open(handle))
            if gt_pix.ndim!=2: raise ValueError("Invalid Penn-Fudan person instance IDs")
            boxes=[]
            for v in np.unique(gt_pix):
                if v==0: continue
                yy,xx=np.where(gt_pix==v)
                boxes.append([int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)])
            samples.append((Path(filename).name,image,boxes))
    return samples

def read_crowd(images,annotations,n):
    records=[json.loads(s) for s in Path(annotations).read_text().splitlines() if s.strip()]
    if len(records)<n: raise RuntimeError("Insufficient annotated original CrowdHuman images")
    samples=[]
    for item in records[:n]:
        frame=Path(images)/(item["ID"]+".jpg")
        image=cv2.imread(str(frame))
        if image is None: raise FileNotFoundError(frame)
        gt=[]
        for x in item.get("gtboxes",[]):
            if x.get("tag")!="person" or x.get("extra",{}).get("ignore",0): continue
            xx,yy,ww,hh=map(float,x["fbox"])
            if min(ww,hh)<=0: continue
            gt.append([xx,yy,xx+ww,yy+hh])
        samples.append((item["ID"],image,gt))
    return samples

def run_domain(domain,samples,model,threshold,output):
    rows=[];gt_count=matched_count=pred_count=0
    for name,image,gt in samples:
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB).copy()
        with torch.inference_mode():
            result=model([torch.from_numpy(rgb).permute(2,0,1).float()/255])[0]
        boxes=result["boxes"].cpu().numpy()
        scores=result["scores"].cpu().numpy()
        labels=result["labels"].cpu().numpy()
        pred=[(box.tolist(),float(score)) for box,score,klass in zip(boxes,scores,labels)
              if int(klass)==1 and float(score)>=threshold]
        matched=match_boxes(pred,gt)
        rows.append({"dataset":domain,"frame":name,"persons":len(gt),
                     "matched":len(matched),"detections":len(pred),
                     "recall":len(matched)/len(gt) if len(gt) else ""})
        gt_count+=len(gt); matched_count+=len(matched);pred_count+=len(pred)
    if not gt_count: raise RuntimeError("No original annotations found: "+domain)
    return rows,{"dataset":domain,"images":len(samples),"persons":gt_count,
                 "matched":matched_count,"detections":pred_count,
                 "local_recall_iou50":matched_count/gt_count}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--crowd-images",type=Path,required=True)
    parser.add_argument("--crowd-annotations",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--scratch",type=Path,default=Path("/tmp/ch5_penn"))
    parser.add_argument("--output",type=Path,default=Path("cross_dataset_results"))
    parser.add_argument("--images",type=int,default=24)
    parser.add_argument("--confidence",type=float,default=.30)
    args=parser.parse_args()
    if not (1<=args.images<=50 and 0<args.confidence<1):
        raise ValueError("Invalid experiment protocol")
    torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=True)
    ckpt=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    model=fasterrcnn_resnet50_fpn(weights=None,weights_backbone=None)
    model.load_state_dict(ckpt,strict=True)
    model.cpu().eval()
    penn=read_penn(get_penn_zip(args.scratch/"PennFudanPed.zip"),args.images)
    crowd=read_crowd(args.crowd_images,args.crowd_annotations,args.images)
    per_penn,result_penn=run_domain("Penn-Fudan",penn,model,args.confidence,args.output)
    per_crowd,result_crowd=run_domain("CrowdHuman",crowd,model,args.confidence,args.output)
    with (args.output/"per_image.csv").open("w",newline="") as fh:
        wr=csv.DictWriter(fh,fieldnames=["dataset","frame","persons","matched","detections","recall"])
        wr.writeheader();wr.writerows(per_penn+per_crowd)
    metrics={
        "status":"actual_model_executed_on_both_original_datasets",
        "model":"torchvision COCO pretrained Faster R-CNN ResNet50-FPN person class",
        "device":"cpu",
        "protocol":"Identical weights, IoU50 one-to-one matching, confidence >= 0.30; original full-body reference boxes",
        "scope":"Exploratory pedestrian domain transfer; NOT official autonomous-driving mAP",
        "results":[result_penn,result_crowd],
        "weight_sha256":hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    }
    (args.output/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9.5,4.3),layout="constrained")
    names=[v["dataset"] for v in metrics["results"]]
    rec=[v["local_recall_iou50"] for v in metrics["results"]]
    bars=axes[0].bar(names,rec,color=["#258199","#b27951"],width=.5)
    for bar,m in zip(bars,metrics["results"]):
        axes[0].text(bar.get_x()+bar.get_width()/2,m["local_recall_iou50"]+.02,
                     f'{m["matched"]}/{m["persons"]}',ha="center")
    axes[0].set(ylabel="Local person recall at IoU ≥ 0.50",ylim=(0,1.1))
    axes[1].scatter(range(len(per_penn)),[float(x["recall"]) for x in per_penn],label="Penn-Fudan",s=22,alpha=.75)
    axes[1].scatter(range(len(per_crowd)),[float(x["recall"]) for x in per_crowd],label="CrowdHuman",s=22,alpha=.75)
    axes[1].set(xlabel="Selected frame index",ylabel="Per-image person recall",ylim=(-.08,1.08))
    axes[1].legend(fontsize=8)
    for ext in ("pdf","png","svg"):
        fig.savefig(args.output/f"cross_dataset_actual_recall.{ext}",dpi=220,bbox_inches="tight")
    print(json.dumps({k:v for k,v in metrics.items() if k!="weight_sha256"},indent=2),flush=True)

if __name__=="__main__":
    main()
