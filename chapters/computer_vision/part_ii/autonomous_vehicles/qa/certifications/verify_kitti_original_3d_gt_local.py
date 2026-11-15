"""Fail-closed independent validation of measured original KITTI 3D GT overlap.

No official KITTI AP, no held-out model generalization, no challenge ignore
handling, and no real-time hardware conclusion can be inferred from 000008.
"""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon

def rotated_corners(box):
    x,y,z,dx,dy,dz,yaw=box
    assert min(dx,dy,dz)>0
    c,s=math.cos(yaw),math.sin(yaw)
    rot=np.array([[c,-s],[s,c]])
    local=np.array([[-dx/2,-dy/2],[dx/2,-dy/2],[dx/2,dy/2],[-dx/2,dy/2]])
    return local@rot.T+[x,y]

def iou3d(a,b):
    pa,pb=Polygon(rotated_corners(a)),Polygon(rotated_corners(b))
    assert pa.is_valid and pb.is_valid
    xy=pa.intersection(pb).area
    z=max(0.,min(a[2]+a[5]/2,b[2]+b[5]/2)-
          max(a[2]-a[5]/2,b[2]-b[5]/2))
    common=xy*z
    va=np.prod(a[3:6]);vb=np.prod(b[3:6])
    return common/(va+vb-common)

def audit(root):
    results=json.loads((root/"measured_3d_iou.json").read_text())
    with (root/"gt_lidar_boxes.csv").open() as f: gt=list(csv.DictReader(f))
    assert results["status"]=="original_KITTI_000008_single_scene_model_vs_original_3D_ground_truth"
    assert len(gt)==results["original_annotated_cars"]==6
    assert sum(int(g["difficulty"])>=0 for g in gt)==results["original_nonnegative_difficulty_cars"]==4
    assert results["actual_pointpillars_car_proposals"]==10
    assert len(results["original_metadata_sha256"])==len(results["original_lidar_sha256"])==64
    matrix=np.asarray(results["iou_matrix"])
    assert matrix.shape==(10,6) and np.isfinite(matrix).all()
    assert np.all((matrix>=0)&(matrix<=1))
    for g in gt:
        b=np.array([float(g[k]) for k in ("x","y","z","dx","dy","dz","yaw")])
        assert abs(iou3d(b,b)-1)<1e-12
        moved=b.copy();moved[0]+=100.
        assert iou3d(b,moved)==0
    for r in results["results"]:
        indices=list(range(6)) if r["scope"]=="all_six_annotated_cars" else [
            i for i,g in enumerate(gt) if int(g["difficulty"])>=0]
        assert r["original_reference_cars"]==len(indices)
        threshold=r["threshold_3d_iou"]
        used=set();independent=[]
        for i in range(10):
            choices=[(matrix[i,j],j) for j in indices if j not in used]
            if choices:
                value,j=max(choices)
                if value>=threshold:
                    used.add(j)
                    independent.append((i,j,float(value)))
        assert len(independent)==r["matched_reference_cars"]
        assert len(independent)==len(r["matches"])
        for expected,actual in zip(r["matches"],independent):
            assert expected["detector_index"]==actual[0] and expected["reference_index"]==actual[1]
            assert abs(expected["iou3d"]-actual[2])<1e-12
        assert abs(r["local_reference_recall"]-len(independent)/len(indices))<1e-12
        assert abs(r["unadjusted_proposal_precision"]-len(independent)/10)<1e-12
    assert all(r["matched_reference_cars"]==r["original_reference_cars"] for r in results["results"])
    return {"result":"PASS","single_scene_gt_car_boxes":6,
            "nonnegative_difficulty_car_boxes":4,
            "actual_predicted_car_boxes":10,
            "matched_car_boxes_at_3D_IoU_0_70":6,
            "scope":"single original KITTI training frame only, NOT official AP",
            "json_sha256":hashlib.sha256((root/"measured_3d_iou.json").read_bytes()).hexdigest()}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("result_directory",type=Path)
    a=p.parse_args()
    print(json.dumps(audit(a.result_directory),indent=2))
