"""CPU-only class-aware camera/LiDAR detection association on original KITTI scene.

Two genuine separately pretrained models:
 - OpenPCDet-compatible PointPillars 3D boxes (torch_pointcloud, KITTI)
 - Torchvision COCO pretrained Faster R-CNN (RGB camera)

This is a calibrated LATE-FUSION experiment. It is NOT a trained joint
BEVFusion, TransFusion or MVX-Net model, nor a KITTI AP benchmark.

Data source: official OpenMMLab MMDetection3D redistributed KITTI scene 000008.
Never create fallback boxes, scores, points, or images.
"""
import argparse
import csv
import hashlib
import io
import json
import pickle
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torch_pointcloud.ops.box3d import nms3d
from torch_pointcloud.utils.data import DataKeys, collate
import torch_pointcloud as tp

URL_BASE="https://raw.githubusercontent.com/open-mmlab/mmdetection3d/main/demo/data/kitti/"


def get_source(name, directory):
    path=directory/name
    if not path.is_file():
        with urllib.request.urlopen(URL_BASE+name,timeout=180) as response:
            payload=response.read()
        if len(payload)<1000:
            raise RuntimeError("Original KITTI sample unavailable: "+name)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(payload)
    return path


def iou2(a,b):
    ax1,ay1,ax2,ay2=map(float,a)
    bx1,by1,bx2,by2=map(float,b)
    w=max(0.,min(ax2,bx2)-max(ax1,bx1))
    h=max(0.,min(ay2,by2)-max(ay1,by1))
    intersection=w*h
    area_a=max(0.,ax2-ax1)*max(0.,ay2-ay1)
    area_b=max(0.,bx2-bx1)*max(0.,by2-by1)
    return intersection/max(1e-12,area_a+area_b-intersection)


def project_3d(box,lidar_to_img,w,h):
    x,y,z,dx,dy,dz,yaw=map(float,box)
    if min(dx,dy,dz)<=0:
        return None
    corners=np.array([[sx*dx/2,sy*dy/2,sz*dz/2]
                      for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)])
    cs,sn=np.cos(yaw),np.sin(yaw)
    rot=np.array([[cs,-sn,0],[sn,cs,0],[0,0,1]])
    pts=corners@rot.T+np.array([x,y,z])
    homogeneous=np.c_[pts,np.ones(8)]
    projected=homogeneous@lidar_to_img.T
    if np.any(projected[:,2]<=0.1):
        return None
    uv=projected[:,:2]/projected[:,2:3]
    x1,y1=np.min(uv,axis=0)
    x2,y2=np.max(uv,axis=0)
    if x2<=0 or y2<=0 or x1>=w or y1>=h:
        return None
    rect=np.array([np.clip(x1,0,w),np.clip(y1,0,h),
                   np.clip(x2,0,w),np.clip(y2,0,h)],float)
    if rect[2]-rect[0]<2 or rect[3]-rect[1]<2:
        return None
    return rect


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path("fusion_evidence"))
    p.add_argument("--scratch",type=Path,default=Path("/tmp/ch5_kitti_fusion"))
    p.add_argument("--camera-score",type=float,default=.25)
    p.add_argument("--lidar-score",type=float,default=.20)
    p.add_argument("--match-iou",type=float,default=.20)
    a=p.parse_args()
    torch.set_num_threads(4)
    a.out.mkdir(parents=True,exist_ok=True)
    cloud_path=get_source("000008.bin",a.scratch)
    image_path=get_source("000008.png",a.scratch)
    calibration_path=get_source("000008.pkl",a.scratch)
    point_cloud=np.fromfile(cloud_path,dtype="<f4").reshape(-1,4)
    image=cv2.imread(str(image_path))
    if not np.isfinite(point_cloud).all() or len(point_cloud)<1000 or image is None:
        raise RuntimeError("Missing or invalid original KITTI inputs")
    height,width=image.shape[:2]
    # Trust restricted to the official public OpenMMLab upstream asset.
    metadata=pickle.loads(calibration_path.read_bytes())
    entry=metadata["data_list"][0]
    lidar_to_image=np.asarray(entry["images"]["CAM2"]["lidar2img"],dtype=float)
    if lidar_to_image.shape!=(4,4) or not np.isfinite(lidar_to_image).all():
        raise RuntimeError("Invalid original KITTI LiDAR-to-CAM2 projection")

    model,model_info=tp.create_model(
        "pointpillars.kitti.openpcdet",task="detection",pretrained=True,return_info=True)
    model=model.cpu().eval()
    sample={"pos":torch.from_numpy(point_cloud[:,:3].copy()),
            "intensity":torch.from_numpy(point_cloud[:,3:4].copy()),
            "frame":"000008"}
    voxels=model_info["transform"](sample)
    batch=collate([voxels],cat_keys=[DataKeys.POS_VOXEL])
    with torch.inference_mode():
        tic=time.perf_counter()
        raw3d=model(batch["voxel"].cpu(),batch["pos_voxel"].cpu(),
                    batch["voxel_num_points"].cpu(),batch["batch_pos_voxel"].cpu())
        decoded=model.decode(raw3d)
        lidar_elapsed=time.perf_counter()-tic
    boxes=decoded["boxes"].cpu()
    scores=decoded["scores"].cpu()
    labels=decoded["labels"].cpu()
    batchidx=decoded["batch"].cpu()
    eligible=(scores>=a.lidar_score).nonzero().squeeze(-1)
    kept=eligible[nms3d(boxes[eligible],scores[eligible],0.01,batch=batchidx[eligible],rotated=True)]
    lidar_classes=model_info["weights"]["classes"]
    lidar=[]
    for i in kept.tolist():
        klass=str(lidar_classes[int(labels[i])])
        rectangle=project_3d(boxes[i].tolist(),lidar_to_image,width,height)
        if rectangle is None:
            continue
        lidar.append({"class":klass,"score":float(scores[i]),
                      "bbox2d":rectangle.tolist(),"box3d":boxes[i].tolist()})
    if not lidar:
        raise RuntimeError("No genuine projected PointPillars detections")

    weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    camera_net=torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    camera_net=camera_net.cpu().eval()
    photo=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    inp=torch.from_numpy(photo.copy()).permute(2,0,1).float()/255.
    with torch.inference_mode():
        tic=time.perf_counter()
        raw2d=camera_net([inp])[0]
        camera_elapsed=time.perf_counter()-tic
    classnames=weights.meta["categories"]
    camera=[]
    for box,score,label in zip(raw2d["boxes"],raw2d["scores"],raw2d["labels"]):
        if float(score)<a.camera_score:
            continue
        klass=str(classnames[int(label)])
        if klass not in ("car","person","bicycle"):
            continue
        camera.append({"class":klass,"score":float(score),
                       "bbox2d":[float(v) for v in box.tolist()]})

    class_map={"Car":"car","Pedestrian":"person","Cyclist":"bicycle"}
    possible=[]
    for i,pred3 in enumerate(lidar):
        for j,pred2 in enumerate(camera):
            if class_map.get(pred3["class"])!=pred2["class"]:
                continue
            overlap=iou2(pred3["bbox2d"],pred2["bbox2d"])
            if overlap>=a.match_iou:
                possible.append((overlap,i,j))
    possible.sort(reverse=True)
    chosen_lidar,chosen_camera,matches=set(),set(),[]
    for overlap,i,j in possible:
        if i in chosen_lidar or j in chosen_camera:
            continue
        chosen_lidar.add(i);chosen_camera.add(j)
        matches.append({"lidar_index":i,"camera_index":j,
                        "class":lidar[i]["class"],"projected_box_iou":overlap,
                        "lidar_score":lidar[i]["score"],
                        "camera_score":camera[j]["score"]})
    report={
        "status":"executed_two_pretrained_detector_models_on_same_original_KITTI_scene",
        "scope":"calibrated geometric late fusion; NOT jointly trained fusion model or official KITTI benchmark",
        "scene":"KITTI 3D detection training 000008",
        "raw_lidar_points":len(point_cloud),
        "image_size":[width,height],
        "model_lidar":"OpenPCDet-compatible pretrained PointPillars",
        "model_camera":"COCO-pretrained Faster R-CNN ResNet50-FPN",
        "lidar_boxes_projecting_into_cam2":len(lidar),
        "camera_boxes_above_threshold":len(camera),
        "matched_cross_sensor_boxes":len(matches),
        "lidar_model_seconds":lidar_elapsed,
        "camera_model_seconds":camera_elapsed,
        "lidar_score_threshold":a.lidar_score,
        "camera_score_threshold":a.camera_score,
        "association_iou_threshold":a.match_iou,
        "source_sha256":{
            "lidar":hashlib.sha256(cloud_path.read_bytes()).hexdigest(),
            "camera":hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "calibration":hashlib.sha256(calibration_path.read_bytes()).hexdigest()
        },
        "lidar_detections":lidar,"camera_detections":camera,"matches":matches
    }
    (a.out/"results.json").write_text(json.dumps(report,indent=2)+"\n")
    with (a.out/"matches.csv").open("w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=["lidar_index","camera_index","class",
                                        "projected_box_iou","lidar_score","camera_score"])
        wr.writeheader();wr.writerows(matches)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,ax=plt.subplots(figsize=(11,4.0),layout="constrained")
    ax.set(xlim=(0,width),ylim=(height,0),
           xlabel="CAM2 horizontal image coordinate (pixels)",
           ylabel="CAM2 vertical image coordinate (pixels)")
    # The figure must display the ORIGINAL KITTI camera image beneath genuine\n    # detector predictions, not merely schematic boxes on blank axes.\n    ax.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),\n              extent=(0,width,height,0),origin="upper",interpolation="nearest")
    for i,d in enumerate(lidar):
        x1,y1,x2,y2=d["bbox2d"]
        ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fc="none",
                               ec="#177f92",lw=1.3,ls="--"))
    for j,d in enumerate(camera):
        x1,y1,x2,y2=d["bbox2d"]
        ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fc="none",
                               ec="#c66c47",lw=1.3))
    for hit in matches:
        i,j=hit["lidar_index"],hit["camera_index"]
        a_rect=lidar[i]["bbox2d"];b_rect=camera[j]["bbox2d"]
        center_a=((a_rect[0]+a_rect[2])/2,(a_rect[1]+a_rect[3])/2)
        center_b=((b_rect[0]+b_rect[2])/2,(b_rect[1]+b_rect[3])/2)
        ax.plot([center_a[0],center_b[0]],[center_a[1],center_b[1]],
                lw=1.6,c="#518859")
    ax.plot([],[],c="#177f92",ls="--",label="LiDAR 3D box projected into camera")
    ax.plot([],[],c="#c66c47",label="Camera Faster R-CNN detection")
    ax.plot([],[],c="#518859",label="IoU-matched cross-sensor association")
    ax.legend(fontsize=8,loc="upper right")
    ax.grid(alpha=.09)\n    ax.set_title("")
    for extension in ("pdf","png","svg"):
        fig.savefig(a.out/f"kitti_calibrated_late_fusion.{extension}",
                    dpi=220,bbox_inches="tight")
    print(json.dumps({k:v for k,v in report.items()
                      if k not in ("source_sha256","lidar_detections","camera_detections","matches")},
                     indent=2),flush=True)

if __name__=="__main__":
    main()
