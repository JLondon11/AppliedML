"""Run a real, externally pretrained TuSimple U-Net on original labeled TuSimple frames.

This is a local segmentation-mask proxy, NOT official TuSimple lane-point accuracy.
Image and model downloads require access to public Hugging Face repositories.
No fabricated images or labels: errors fail closed.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import re
import zipfile
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import cv2
import keras
import numpy as np
from huggingface_hub import hf_hub_download

MODEL_REPO = "Tilak1812/lane-detection-unet-tusimple"
DATA_REPO = "AhmedNasr5804/Lane-and-eye-detection-dataset"


def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def find_annotation_records(z):
    names=[n for n in z.namelist() if n.lower().endswith((".json",".jsonl"))]
    records=[]
    for name in names:
        # Not every archive JSON is a TuSimple annotation.
        if not (("label" in name.lower()) or ("task" in name.lower()) or ("train" in name.lower())):
            continue
        try:
            with z.open(name) as fh:
                text=fh.read().decode("utf-8")
            if text.lstrip().startswith("["):
                samples=json.loads(text)
            else:
                samples=[json.loads(line) for line in text.splitlines() if line.strip()]
            for r in samples:
                if isinstance(r,dict) and "raw_file" in r and "lanes" in r and "h_samples" in r:
                    records.append(r)
        except (ValueError,UnicodeDecodeError,KeyError,TypeError):
            continue
    # keyed original source paths; fail if different annotations reuse one path
    out={}
    for r in records:
        key=str(r["raw_file"]).strip().replace("\\","/")
        if key and key not in out:
            out[key]=r
    return out


def original_frame(z,members,path):
    name=path.replace("\\","/").lstrip("/")
    keys=[name,"tusimple/"+name,"train_set/"+name,"test_set/"+name]
    for key in keys:
        if key in members:
            return z.read(members[key])
    suffix="/".join(name.split("/")[-3:])
    matches=[member for member in members if member.endswith("/"+name) or member.endswith("/"+suffix)]
    if len(matches)==1:
        return z.read(members[matches[0]])
    examples=[v for v in members if v.lower().endswith((".jpg",".jpeg",".png"))][:12]
    raise FileNotFoundError("Annotated original frame not found uniquely in archive: "+name+"; suffix="+suffix+"; matches="+repr(matches[:8])+"; first image entries="+repr(examples))


def gt_mask(record,hw,thickness=4):
    h,w=hw
    gt=np.zeros((h,w),np.uint8)
    yy=record["h_samples"]
    for lane in record["lanes"]:
        if len(lane)!=len(yy):
            raise ValueError("Invalid TuSimple annotation lane length")
        pts=np.array([(int(x),int(y)) for x,y in zip(lane,yy)
                      if 0<=int(x)<w and 0<=int(y)<h],np.int32)
        if len(pts)>1:
            cv2.polylines(gt,[pts],False,1,thickness=thickness)
    return gt


def scores(pred,gt):
    p=np.asarray(pred,bool)
    g=np.asarray(gt,bool)
    if p.shape!=g.shape:
        raise ValueError("Incompatible mask shapes")
    tp=int(np.count_nonzero(p&g))
    fp=int(np.count_nonzero(p&~g))
    fn=int(np.count_nonzero(~p&g))
    return tp,fp,fn,tp/max(1,tp+fp),tp/max(1,tp+fn),tp/max(1,tp+fp+fn),2*tp/max(1,2*tp+fp+fn)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=Path("tusimple_results"))
    parser.add_argument("--images",type=int,default=12)
    parser.add_argument("--seed",type=int,default=20261106)
    parser.add_argument("--threshold",type=float,default=.5)
    parser.add_argument("--archive",type=Path,default=None)
    parser.add_argument("--checkpoint",type=Path,default=None)
    args=parser.parse_args()
    if not 1<=args.images<=200 or not 0<args.threshold<1:
        parser.error("Invalid subset size or mask threshold")
    import torch
    torch.set_num_threads(4)
    args.out.mkdir(parents=True,exist_ok=True)
    archive=args.archive or Path(hf_hub_download(DATA_REPO,"tusimple.zip",repo_type="dataset"))
    checkpoint=args.checkpoint or Path(hf_hub_download(MODEL_REPO,"best_model.keras"))
    print("Source archive",archive,"bytes",archive.stat().st_size,flush=True)
    print("Model file",checkpoint,"bytes",checkpoint.stat().st_size,flush=True)
    if checkpoint.stat().st_size<100000:
        raise RuntimeError("Pretrained model file too small")
    model=keras.saving.load_model(str(checkpoint),compile=False,safe_mode=True)
    if len(model.inputs)!=1 or len(model.outputs)!=1:
        raise ValueError("Require one-input, one-output pretrained mask model")
    inshape=model.inputs[0].shape
    if len(inshape)!=4 or inshape[-1]!=3 or not isinstance(inshape[1],int) or not isinstance(inshape[2],int):
        raise ValueError("Expected fixed-size NHWC RGB model input")
    height,width=inshape[1:3]
    outshape=model.outputs[0].shape
    print("Model shapes",str(inshape),str(outshape),flush=True)
    rows=[]
    with zipfile.ZipFile(archive) as z:
        members={n:n for n in z.namelist() if not n.endswith("/")}
        annotation=find_annotation_records(z)
        print("Original annotated unique frames",len(annotation),"archive entries",len(members),flush=True)
        if len(annotation)<args.images:
            raise RuntimeError("Dataset archive lacks required TuSimple annotations")
        chosen=random.Random(args.seed).sample(sorted(annotation),args.images)
        for index,raw_name in enumerate(chosen):
            blob=original_frame(z,members,raw_name)
            img=cv2.imdecode(np.frombuffer(blob,np.uint8),cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Unparseable original dataset image: "+raw_name)
            orig_h,orig_w=img.shape[:2]
            if orig_h<300 or orig_w<500:
                raise ValueError("Unexpected original TuSimple resolution")
            gt=gt_mask(annotation[raw_name],img.shape[:2])
            if gt.sum()==0:
                raise ValueError("No valid ground-truth lane coordinates: "+raw_name)
            rgb=cv2.cvtColor(cv2.resize(img,(width,height)),cv2.COLOR_BGR2RGB)
            input_array=(rgb.astype(np.float32)/255.0)[None]
            predicted=np.asarray(model.predict(input_array,verbose=0))
            if predicted.ndim!=4 or predicted.shape[0]!=1 or predicted.shape[-1]!=1:
                raise ValueError("Unknown mask output; model-specific decoder required: "+str(predicted.shape))
            heat=predicted[0,:,:,0]
            if not np.isfinite(heat).all():
                raise ValueError("Nonfinite neural-network prediction")
            if heat.min()<0 or heat.max()>1:
                heat=1/(1+np.exp(-np.clip(heat,-40,40)))
            heat=cv2.resize(heat,(orig_w,orig_h),interpolation=cv2.INTER_LINEAR)
            mask=(heat>=args.threshold).astype(np.uint8)
            tp,fp,fn,prec,recall,iou,dice=scores(mask,gt)
            rows.append([raw_name,orig_w,orig_h,int(gt.sum()),int(mask.sum()),tp,fp,fn,prec,recall,iou,dice,
                         hashlib.sha256(blob).hexdigest()])
            if index<6:
                canvas=img.copy()
                canvas[gt.astype(bool)]=[40,180,60]
                canvas[mask.astype(bool)]=[25,80,230]
                both=gt.astype(bool)&mask.astype(bool)
                canvas[both]=[250,220,60]
                cv2.imwrite(str(args.out/f"prediction_{index:03d}.png"),canvas)
            print(f"scored {index+1}/{len(chosen)} {raw_name} IoU={iou:.4f} Dice={dice:.4f}",flush=True)
    fields=["source_frame","width","height","gt_pixels","predicted_pixels","tp","fp","fn","precision","recall","iou","dice","image_sha256"]
    with (args.out/"per_frame.csv").open("w",newline="") as f:
        wr=csv.writer(f);wr.writerow(fields);wr.writerows(rows)
    total=np.sum(np.asarray([r[5:8] for r in rows],np.int64),axis=0)
    tp,fp,fn=map(int,total)
    report={"status":"executed_pretrained_model_on_original_dataset",
            "model":MODEL_REPO,"dataset":DATA_REPO,"device":"cpu",
            "evaluation":"local 4-pixel polyline rasterization proxy (NOT official TuSimple lane-point accuracy)",
            "images":len(rows),"seed":args.seed,"mask_threshold":args.threshold,
            "micro_precision":tp/max(1,tp+fp),"micro_recall":tp/max(1,tp+fn),
            "micro_iou":tp/max(1,tp+fp+fn),"micro_dice":2*tp/max(1,2*tp+fp+fn),
            "macro_iou":float(np.mean([r[10] for r in rows])),
            "archive_sha256":sha256(archive),"checkpoint_sha256":sha256(checkpoint)}
    (args.out/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if not k.endswith("sha256")},indent=2),flush=True)


if __name__=="__main__":
    main()
