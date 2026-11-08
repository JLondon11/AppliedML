"""Independently validate *executed* single-scene KITTI PointPillars and fusion artifacts.

No KITTI mAP, 3D reference matching, statistical generalization, or timing
certification can be inferred from these consistency tests.
Usage: python verify_kitti_real_execution.py pointpillars.zip fusion.zip
"""
import argparse,csv,hashlib,io,json,math,zipfile
from collections import Counter
from pathlib import Path
import numpy as np

def iou(a,b):
    ax1,ay1,ax2,ay2=map(float,a);bx1,by1,bx2,by2=map(float,b)
    w=max(0,min(ax2,bx2)-max(ax1,bx1))
    h=max(0,min(ay2,by2)-max(ay1,by1))
    inter=w*h
    aa=(ax2-ax1)*(ay2-ay1);bb=(bx2-bx1)*(by2-by1)
    assert aa>0 and bb>0
    return inter/(aa+bb-inter)

def verify(pointpillars,fusion_archive):
    with zipfile.ZipFile(pointpillars) as a,zipfile.ZipFile(fusion_archive) as b:
        p=json.loads(a.read("metrics.json"))
        f=json.loads(b.read("results.json"))
        csv_matches=list(csv.DictReader(io.StringIO(b.read("matches.csv").decode())))
        for archive,base in ((a,"pointpillars_kitti_000008_cpu"),(b,"kitti_calibrated_late_fusion")):
            for ext in ("png","pdf","svg"):
                assert archive.getinfo(base+"."+ext).file_size>8000
    assert p["status"]=="executed_original_pretrained_pointpillars_cpu"
    assert f["status"]=="executed_two_pretrained_detector_models_on_same_original_KITTI_scene"
    assert p["model"]=="pointpillars.kitti.openpcdet"
    assert p["source_points"]==f["raw_lidar_points"]==17238
    assert f["image_size"]==[1242,375]
    assert p["input_sha256"]==f["source_sha256"]["lidar"]
    for v in f["source_sha256"].values():assert len(v)==64
    assert p["score_threshold"]==f["lidar_score_threshold"]==.20
    assert p["rotated_nms_threshold"]==.01
    assert f["association_iou_threshold"]==.20
    assert p["detected_boxes"]==len(p["detections"])==18
    assert f["lidar_boxes_projecting_into_cam2"]==len(f["lidar_detections"])==18
    assert f["camera_boxes_above_threshold"]==len(f["camera_detections"])==15
    assert f["matched_cross_sensor_boxes"]==len(f["matches"])==len(csv_matches)==9
    assert min(p["wall_clock_model_seconds"],f["lidar_model_seconds"],f["camera_model_seconds"])>0
    counts=Counter()
    for item in p["detections"]:
        box=np.asarray(item["xyz_dxdydz_yaw"],float)
        assert box.shape==(7,) and np.isfinite(box).all()
        assert np.all(box[3:6]>0) and max(box[3:6])<30
        assert item["class"] in ("Car","Pedestrian","Cyclist")
        assert .20<=item["score"]<=1
        counts[item["class"]]+=1
    seen=set()
    for l in f["lidar_detections"]:
        b=np.asarray(l["box3d"],float)
        rect=np.asarray(l["bbox2d"],float)
        assert np.isfinite(rect).all() and 0<=rect[0]<rect[2]<=1242 and 0<=rect[1]<rect[3]<=375
        matching=[i for i,x in enumerate(p["detections"])
                  if x["class"]==l["class"] and abs(x["score"]-l["score"])<1e-7
                  and np.allclose(x["xyz_dxdydz_yaw"],b,atol=1e-6,rtol=0)]
        assert len(matching)==1 and matching[0] not in seen
        seen.add(matching[0])
    assert len(seen)==18
    for c in f["camera_detections"]:
        r=c["bbox2d"]
        assert c["class"] in ("car","person","bicycle")
        assert c["score"]>=.25
        assert 0<=r[0]<r[2]<=1242 and 0<=r[1]<r[3]<=375
    mapping={"Car":"car","Pedestrian":"person","Cyclist":"bicycle"}
    possible=[]
    for i,l in enumerate(f["lidar_detections"]):
        for j,c in enumerate(f["camera_detections"]):
            if mapping[l["class"]]==c["class"]:
                v=iou(l["bbox2d"],c["bbox2d"])
                if v>=.20:possible.append((v,i,j))
    possible.sort(reverse=True)
    usedl,usedc=set(),set();independent=[]
    for v,i,j in possible:
        if i not in usedl and j not in usedc:
            usedl.add(i);usedc.add(j);independent.append((i,j,v))
    assert len(independent)==9
    for expected,actual,csvrow in zip(f["matches"],independent,csv_matches):
        i,j,v=actual
        assert (expected["lidar_index"],expected["camera_index"])==(i,j)
        assert abs(expected["projected_box_iou"]-v)<1e-12
        assert expected["class"]==f["lidar_detections"][i]["class"]
        for key in ("lidar_index","camera_index"):
            assert int(csvrow[key])==expected[key]
        for key in ("projected_box_iou","lidar_score","camera_score"):
            assert abs(float(csvrow[key])-expected[key])<1e-12
    return {"result":"PASS","scope":"single original KITTI scene; NO official KITTI AP",
            "lidar_points":17238,"oriented_3d_detections":18,
            "camera_detections":15,"independent_associations":9,
            "class_counts":dict(counts),
            "pointpillars_zip_sha256":hashlib.sha256(Path(pointpillars).read_bytes()).hexdigest(),
            "fusion_zip_sha256":hashlib.sha256(Path(fusion_archive).read_bytes()).hexdigest()}

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("pointpillars");parser.add_argument("fusion")
    a=parser.parse_args()
    print(json.dumps(verify(a.pointpillars,a.fusion),indent=2))
