"""Genuine pretrained PointPillars inference on official KITTI LiDAR frame 000008.

This is a single-frame smoke/inference study, not a KITTI mAP benchmark.
No synthetic boxes or fixed result values are permitted.
"""
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

# Official upstream MMDetection3D redistributes original KITTI sample from frame 000008.
SOURCE = "https://raw.githubusercontent.com/open-mmlab/mmdetection3d/main/demo/data/kitti/000008.bin"

def load_source(out):
    target=out/"KITTI_000008_original_velodyne.bin"
    if not target.exists():
        with urllib.request.urlopen(SOURCE,timeout=180) as req:
            data=req.read()
        if len(data)<100_000 or len(data)%16:
            raise RuntimeError("Original KITTI frame missing or invalid")
        target.write_bytes(data)
    points=np.fromfile(target,dtype="<f4")
    if points.size%4:
        raise RuntimeError("KITTI file is not XYZI")
    points=points.reshape(-1,4)
    if not np.isfinite(points).all() or len(points)<20000:
        raise RuntimeError("KITTI data absent or corrupt")
    return target,points

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=Path("pointpillars_results"))
    parser.add_argument("--min-score",type=float,default=.20)
    parser.add_argument("--nms-iou",type=float,default=.01)
    a=parser.parse_args()
    import torch
    import torch_pointcloud as tp
    from torch_pointcloud.ops.box3d import nms3d
    from torch_pointcloud.utils.data import DataKeys,collate

    torch.set_num_threads(4)
    a.out.mkdir(exist_ok=True,parents=True)
    file,points=load_source(a.out)
    print("Loaded original KITTI frame:",len(points),"points",flush=True)
    model,info=tp.create_model("pointpillars.kitti.openpcdet",
                               task="detection",pretrained=True,return_info=True)
    model=model.cpu().eval()
    sample={"pos":torch.from_numpy(points[:,:3].copy()),
            "intensity":torch.from_numpy(points[:,3:4].copy()),
            "frame":"000008"}
    sample=info["transform"](sample)
    data=collate([sample],cat_keys=[DataKeys.POS_VOXEL])
    with torch.inference_mode():
        started=time.perf_counter()
        prediction=model(data["voxel"].cpu(),
                         data["pos_voxel"].cpu(),
                         data["voxel_num_points"].cpu(),
                         data["batch_pos_voxel"].cpu())
        decoded=model.decode(prediction)
        elapsed=time.perf_counter()-started
    boxes=decoded["boxes"].cpu()
    scores=decoded["scores"].cpu()
    labels=decoded["labels"].cpu()
    batch=decoded["batch"].cpu()
    if boxes.ndim!=2 or boxes.shape[1]!=7 or not torch.isfinite(boxes).all():
        raise RuntimeError("Pretrained PointPillars produced malformed boxes")
    good=(scores>a.min_score).nonzero().squeeze(-1)
    kept=good[nms3d(boxes[good],scores[good],a.nms_iou,batch=batch[good],rotated=True)]
    classes=info["weights"]["classes"]
    detections=[]
    for idx in kept.tolist():
        detections.append({
            "class":str(classes[int(labels[idx])]),
            "score":float(scores[idx]),
            "xyz_dxdydz_yaw":[float(x) for x in boxes[idx].tolist()]
        })
    results={
        "status":"executed_original_pretrained_pointpillars_cpu",
        "dataset":"Original KITTI detection training scene 000008; MMDetection3D demo mirror",
        "source":SOURCE,
        "model":"pointpillars.kitti.openpcdet",
        "device":"cpu",
        "metric_scope":"single original scene; NO official KITTI AP",
        "source_points":len(points),
        "score_threshold":a.min_score,
        "rotated_nms_threshold":a.nms_iou,
        "detected_boxes":len(detections),
        "wall_clock_model_seconds":elapsed,
        "input_sha256":hashlib.sha256(file.read_bytes()).hexdigest(),
        "detections":detections
    }
    (a.out/"metrics.json").write_text(json.dumps(results,indent=2)+"\n")
    # Scientific BEV figure is generated from actual measured LiDAR points and predicted boxes.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    fig,ax=plt.subplots(figsize=(9.0,6.3),layout="constrained")
    valid=(points[:,0]>0)&(points[:,0]<70)&(abs(points[:,1])<40)
    xy=points[valid,:2]
    ax.scatter(xy[::5,0],xy[::5,1],s=.15,c="#7c8997",alpha=.50,rasterized=True)
    palette={"Car":"#c56a47","Pedestrian":"#5b76aa","Cyclist":"#4f8b76"}
    for det in detections:
        x,y,z,dx,dy,dz,yaw=det["xyz_dxdydz_yaw"]
        corners=np.array([[-dx/2,-dy/2],[dx/2,-dy/2],[dx/2,dy/2],[-dx/2,dy/2]])
        rot=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
        poly=corners@rot.T+np.array([x,y])
        ax.add_patch(Polygon(poly,closed=True,fill=False,
                             edgecolor=palette.get(det["class"],"#d4a44c"),linewidth=1.25))
    ax.set(xlim=(0,70),ylim=(-40,40),xlabel="Forward distance x (m)",
           ylabel="Lateral distance y (m)")
    ax.set_aspect("equal");ax.grid(alpha=.13)
    for ext in ("png","pdf","svg"):
        fig.savefig(a.out/f"pointpillars_kitti_000008_cpu.{ext}",dpi=210,bbox_inches="tight")
    print(json.dumps({k:v for k,v in results.items() if k!="detections"},indent=2),flush=True)

if __name__=="__main__":
    main()
