"""Full 170-image original Penn-Fudan paired pretrained-detector comparison.

Both torchvision COCO-pretrained detector architectures are actually executed,
on identical full original images, with identical person IoU50 and threshold
protocol. Report local VOC-style AP50 and image-cluster bootstrap of the
difference in matched-person recall. Not an occlusion challenge benchmark.
"""
import argparse,csv,hashlib,json,sys,urllib.request,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
ROOT=Path(__file__).resolve().parents[2]/"case_studies"/"occluded_pedestrian_detection"
sys.path.insert(0,str(ROOT))
from evaluate_pennfudan_pretrained import ground_truth,local_ap50,iou_matrix

URL="https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"

def download(path):
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        with urllib.request.urlopen(URL,timeout=180) as src,path.open("wb") as dest:
            while True:
                block=src.read(1<<20)
                if not block:break
                dest.write(block)
    if path.stat().st_size<1_000_000:raise RuntimeError("Not original Penn-Fudan zip")
    return path

def greedy_match(boxes,scores,gt):
    used=set()
    hits=0
    for i in np.argsort(-np.asarray(scores)):
        overlap=iou_matrix([boxes[i]],gt)[0]
        if len(overlap):
            for j in used:overlap[j]=-1
            if float(overlap.max())>=.5:
                j=int(np.argmax(overlap));used.add(j);hits+=1
    return hits

def run_model(model,images,weights,threshold):
    model=model.cpu().eval()
    result=[];per_image=[]
    with torch.inference_mode():
        for name,img,gt in images:
            pred=model([weights.transforms()(img)])[0]
            keep=(pred["labels"].cpu().numpy()==1)&(pred["scores"].cpu().numpy()>=threshold)
            boxes=pred["boxes"].cpu().numpy()[keep]
            scores=pred["scores"].cpu().numpy()[keep]
            result.append({"name":name,"gt":gt,"pred":boxes,"scores":scores})
            hits=greedy_match(boxes,scores,gt)
            per_image.append({"image":name,"gt":len(gt),"matched":hits,
                              "detections":len(boxes),"recall":hits/max(1,len(gt))})
    return result,per_image

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path("pennfudan_two_model_results"))
    p.add_argument("--source",type=Path,default=Path("/tmp/PennFudanPed.zip"))
    p.add_argument("--score",type=float,default=.01)
    a=p.parse_args()
    if not 0<=a.score<1:raise ValueError("Bad confidence threshold")
    torch.set_num_threads(4)
    a.out.mkdir(parents=True,exist_ok=True)
    z=download(a.source)
    with zipfile.ZipFile(z) as archive:
        files=sorted(n for n in archive.namelist() if n.startswith("PennFudanPed/PNGImages/") and n.endswith(".png"))
        if len(files)!=170:raise RuntimeError(f"Expected full 170 images, found {len(files)}")
        images=[];hashes=[]
        for file in files:
            n=Path(file).name
            m="PennFudanPed/PedMasks/"+Path(file).stem+"_mask.png"
            if m not in archive.namelist():raise RuntimeError("Missing original annotation "+m)
            b=archive.read(file)
            img=Image.open(__import__("io").BytesIO(b)).convert("RGB")
            mask=a.out/"temporary_mask.png"
            mask.write_bytes(archive.read(m))
            gt=ground_truth(mask)
            mask.unlink()
            images.append((n,img,gt))
            hashes.append(hashlib.sha256(b).hexdigest())
    if len(set(hashes))!=170:raise RuntimeError("Duplicate original images")
    models=[
       ("MobileNetV3-FPN",fasterrcnn_mobilenet_v3_large_320_fpn,FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT),
       ("ResNet50-FPN",fasterrcnn_resnet50_fpn,FasterRCNN_ResNet50_FPN_Weights.DEFAULT),
    ]
    allrows=[];summary=[]
    for name,construct,weights in models:
        net=construct(weights=weights)
        records,rows=run_model(net,images,weights,a.score)
        metrics=local_ap50(records)
        summary.append({"model":name,**metrics,"evaluated_images":len(rows),
                        "checkpoint_url":weights.url})
        allrows.extend([{"model":name,**r} for r in rows])
        print(json.dumps(summary[-1]),flush=True)
        del net
    with (a.out/"per_image.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["model","image","gt","matched","detections","recall"])
        writer.writeheader();writer.writerows(allrows)
    # paired image bootstrap: descriptive recall difference on this fixed 170-image set
    grouped={model["model"]:{r["image"]:r for r in allrows if r["model"]==model["model"]} for model in summary}
    names=sorted(grouped["MobileNetV3-FPN"])
    m=grouped["MobileNetV3-FPN"];r=grouped["ResNet50-FPN"]
    rng=np.random.default_rng(20261108)
    differences=[]
    for _ in range(3000):
        sample=rng.choice(names,len(names),replace=True)
        ref=sum(m[k]["gt"] for k in sample)
        if ref:
            differences.append((sum(r[k]["matched"] for k in sample)-
                                sum(m[k]["matched"] for k in sample))/ref)
    interval=list(map(float,np.quantile(differences,[.025,.975])))
    report={"dataset":"Original complete PennFudanPed instance-mask dataset",
        "source":URL,"images":170,"reference_persons":int(summary[0]["ground_truth_persons"]),
        "confidence_threshold":a.score,"matching":"One-to-one 2D person IoU>=.5",
        "model_checkpoint_provenance":"official torchvision COCO-trained weights",
        "model_results":summary,"paired_recall_difference_resnet_minus_mobilenet_ci95":interval,
        "bootstrap_unit":"Original image; 3000 paired bootstrap draws, seed=20261108",
        "scope":"Local full 170-image detection evaluation; NOT CrowdHuman occlusion or official COCO AP"}
    (a.out/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    cats=[m["model"] for m in summary]
    ax[0].bar(cats,[m["local_ap50"] for m in summary],color=["#3b8593","#bc7755"])
    ax[0].set(ylabel="Local AP50",ylim=(0,1.08))
    ax[1].bar(cats,[m["recall_at_last_prediction"] for m in summary],color=["#3b8593","#bc7755"])
    ax[1].set(ylabel="Person recall at score >= 0.01",ylim=(0,1.08))
    for axes in ax:
        for label in axes.get_xticklabels():label.set_rotation(9)
        axes.grid(alpha=.12,axis="y")
    for ext in ("pdf","png","svg"):
        fig.savefig(a.out/f"pennfudan_full170_paired_model_comparison.{ext}",dpi=200,bbox_inches="tight")
    print("PASSED full 170 original-image paired-model experiment",flush=True)
if __name__=="__main__":main()
