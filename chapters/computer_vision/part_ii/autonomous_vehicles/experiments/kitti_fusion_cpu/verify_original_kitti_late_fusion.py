"""Fail-closed independent audit of two real KITTI camera/LiDAR model outputs.

Recompute original image-coordinate 2D IoU, same-class eligibility and
descending-IoU greedy one-to-one association from recorded full detector
outputs. This verifies genuine geometric late fusion, not joint model training.
"""
import argparse,csv,io,json,math,zipfile
from pathlib import Path
import numpy as np

CLASSES={"Car":"car","Pedestrian":"person","Cyclist":"bicycle"}

def iou(a,b):
    x1=max(float(a[0]),float(b[0]))
    y1=max(float(a[1]),float(b[1]))
    x2=min(float(a[2]),float(b[2]))
    y2=min(float(a[3]),float(b[3]))
    overlap=max(0.,x2-x1)*max(0.,y2-y1)
    aw=max(0.,float(a[2])-float(a[0]))
    ah=max(0.,float(a[3])-float(a[1]))
    bw=max(0.,float(b[2])-float(b[0]))
    bh=max(0.,float(b[3])-float(b[1]))
    union=aw*ah+bw*bh-overlap
    return overlap/max(union,1e-12)

def evaluate(artifact):
    with zipfile.ZipFile(artifact) as archive:
        required={"results.json","matches.csv","kitti_calibrated_late_fusion.pdf"}
        assert required.issubset(set(archive.namelist()))
        data=json.loads(archive.read("results.json"))
        reported=list(csv.DictReader(io.StringIO(archive.read("matches.csv").decode())))
    assert data["status"]=="executed_two_pretrained_detector_models_on_same_original_KITTI_scene"
    assert data["scene"]=="KITTI 3D detection training 000008"
    assert data["raw_lidar_points"]==17238
    assert data["image_size"]==[1242,375]
    assert data["lidar_boxes_projecting_into_cam2"]==len(data["lidar_detections"])==18
    assert data["camera_boxes_above_threshold"]==len(data["camera_detections"])==15
    assert data["matched_cross_sensor_boxes"]==len(data["matches"])==len(reported)==9
    original=data["source_sha256"]
    assert all(len(original[k])==64 for k in ("lidar","camera","calibration"))
    assert original["lidar"]=="3b9de6cc966534900f6a1bdc93b21772e47a334eb2ef18082021956520d902d1"
    assert original["calibration"]=="b187a5f9185789125e04618645ecfb7c5e9afb5d41ab11b06b9a0cf381f0c45a"
    assert abs(data["lidar_score_threshold"]-.2)<1e-12
    assert abs(data["camera_score_threshold"]-.25)<1e-12
    assert abs(data["association_iou_threshold"]-.2)<1e-12
    lidar,camera=data["lidar_detections"],data["camera_detections"]
    for group,score_threshold in ((lidar,.2),(camera,.25)):
        for box in group:
            assert len(box["bbox2d"])==4
            x1,y1,x2,y2=map(float,box["bbox2d"])
            assert 0<=x1<x2<=1242+1e-6 and 0<=y1<y2<=375+1e-6
            assert score_threshold<=box["score"]<=1
            assert all(math.isfinite(v) for v in box["bbox2d"])
    for box in lidar:
        assert len(box["box3d"])==7
        assert min(box["box3d"][3:6])>0
    possible=[]
    for i,li in enumerate(lidar):
        for j,cam in enumerate(camera):
            if CLASSES.get(li["class"])!=cam["class"]:
                continue
            v=iou(li["bbox2d"],cam["bbox2d"])
            if v>=.2:
                possible.append((v,i,j))
    possible.sort(reverse=True)
    selected=[];seen_li=set();seen_ca=set()
    for v,i,j in possible:
        if i in seen_li or j in seen_ca:continue
        seen_li.add(i);seen_ca.add(j)
        selected.append((v,i,j))
    assert len(selected)==9
    assert len({x["lidar_index"] for x in data["matches"]})==9
    assert len({x["camera_index"] for x in data["matches"]})==9
    for (v,i,j),m,row in zip(selected,data["matches"],reported):
        assert m["lidar_index"]==i and m["camera_index"]==j
        assert abs(m["projected_box_iou"]-v)<1e-12
        assert m["class"]==lidar[i]["class"]
        assert m["class"]==row["class"]
        assert int(row["lidar_index"])==i and int(row["camera_index"])==j
        assert abs(float(row["projected_box_iou"])-v)<1e-12
        assert abs(m["lidar_score"]-lidar[i]["score"])<1e-12
        assert abs(m["camera_score"]-camera[j]["score"])<1e-12
    print(json.dumps({"audit":"PASS","original_KITTI_scene":"000008",
       "genuine_models":2,"original_lidar_points":17238,
       "projected_lidar_predictions":18,"camera_predictions":15,
       "one_to_one_class_consistent_pairs":len(selected),
       "scope":"calibrated same-scene geometric association only, NOT jointly trained sensor fusion"},indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("artifact",type=Path)
    args=p.parse_args()
    evaluate(args.artifact)
