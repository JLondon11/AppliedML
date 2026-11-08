"""Fail-closed independent review of executed PointPillars vs original KITTI 3D GT.

Input: actual workflow artifact zip with measured_3d_iou.json. Tests exact
3D overlap symmetries, threshold monotonicity, and real match consistency.
"""
import argparse,io,json,math,zipfile
from pathlib import Path
import numpy as np

from evaluate_single_scene_original_3d_gt import iou3d

def main(artifact):
    with zipfile.ZipFile(artifact) as z:
        required={"measured_3d_iou.json","gt_lidar_boxes.csv",
                  "kitti_real_3d_gt_vs_pointpillars.pdf"}
        assert required.issubset(set(z.namelist()))
        report=json.loads(z.read("measured_3d_iou.json"))
    assert report["sample"]=="000008"
    assert report["original_annotated_cars"]==6
    assert report["original_nonnegative_difficulty_cars"]==4
    assert report["actual_pointpillars_car_proposals"]==10
    assert len(report["original_lidar_sha256"])==64
    assert len(report["original_metadata_sha256"])==64
    mat=np.asarray(report["iou_matrix"],float)
    assert mat.shape==(10,6) and np.isfinite(mat).all()
    assert (mat>=0).all() and (mat<=1).all()
    # Mathematically independent basic geometry assertions.
    box=np.array([10.,-1.,.6,4.,1.8,1.6,.4])
    shifted=box.copy();shifted[0]+=30
    rotating=box.copy();rotating[6]+=2*math.pi
    assert abs(iou3d(box,box)-1)<1e-12
    assert abs(iou3d(box,rotating)-1)<1e-12
    assert iou3d(box,shifted)==0
    another=np.array([11.,-1.1,.65,3.9,2.0,1.4,-.3])
    assert abs(iou3d(box,another)-iou3d(another,box))<1e-12
    # Detections match original reference GT only once per threshold.
    results=report["results"]
    assert len(results)==6
    for scope,count in [("all_six_annotated_cars",6),
                        ("four_nonnegative_difficulty_cars",4)]:
        entries=sorted((r for r in results if r["scope"]==scope),
                       key=lambda r:r["threshold_3d_iou"])
        assert [r["threshold_3d_iou"] for r in entries]==[.25,.5,.7]
        assert [r["matched_reference_cars"] for r in entries]==[count]*3
        for r in entries:
            assert r["original_reference_cars"]==count
            assert r["pretrained_proposed_cars"]==10
            assert r["matched_reference_cars"]==len(r["matches"])
            assert abs(r["local_reference_recall"]-1)<1e-12
            assert abs(r["unadjusted_proposal_precision"]-count/10)<1e-12
            gts=[m["reference_index"] for m in r["matches"]]
            preds=[m["detector_index"] for m in r["matches"]]
            assert len(gts)==len(set(gts))
            assert len(preds)==len(set(preds))
            for m in r["matches"]:
                a,b=m["detector_index"],m["reference_index"]
                assert 0<=a<10 and 0<=b<6
                assert abs(m["iou3d"]-mat[a,b])<1e-12
                assert m["iou3d"]+1e-12>=r["threshold_3d_iou"]
        assert all(entries[k]["matched_reference_cars"]>=entries[k+1]["matched_reference_cars"]
                   for k in range(2))
    print(json.dumps({"audit":"PASS","original_scene":"000008","original_3d_car_boxes":6,
                      "original_car_proposals":10,"matched_at_iou70":6,
                      "reported_eval_scope":"one training scene; NOT official KITTI mAP"},
                     indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("artifact",type=Path)
    args=p.parse_args();main(args.artifact)
