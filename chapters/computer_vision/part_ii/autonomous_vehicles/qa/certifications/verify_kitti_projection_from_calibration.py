"""Independent reproduction of LiDAR 3D-to-CAM2 box projection on original KITTI.

Only unpickle calibration from the trusted upstream OpenMMLab demo resource.
This tests calibration and projection consistency, NOT KITTI detection accuracy.
"""
import argparse,hashlib,json,pickle,zipfile
from pathlib import Path
import numpy as np

def project(box,matrix,width,height):
    center=np.asarray(box[:3],dtype=float)
    dims=np.asarray(box[3:6],dtype=float)
    yaw=float(box[6])
    assert np.all(dims>0) and np.isfinite(box).all()
    corners=[]
    for sx in (-1.,1.):
        for sy in (-1.,1.):
            for sz in (-1.,1.):
                local=np.array([sx*dims[0]/2,sy*dims[1]/2,sz*dims[2]/2])
                x=local[0]*np.cos(yaw)-local[1]*np.sin(yaw)
                y=local[0]*np.sin(yaw)+local[1]*np.cos(yaw)
                corners.append([center[0]+x,center[1]+y,center[2]+local[2],1.])
    vectors=np.asarray(corners)@matrix.T
    if np.any(vectors[:,2]<=.1):return None
    coords=vectors[:,:2]/vectors[:,2,None]
    xmin,ymin=coords.min(axis=0);xmax,ymax=coords.max(axis=0)
    if xmax<=0 or ymax<=0 or xmin>=width or ymin>=height:return None
    rect=np.array([np.clip(xmin,0,width),np.clip(ymin,0,height),
                   np.clip(xmax,0,width),np.clip(ymax,0,height)])
    if rect[2]-rect[0]<2 or rect[3]-rect[1]<2:return None
    return rect

def verify(calibration,archive):
    with calibration.open("rb") as f:metadata=pickle.load(f)
    entry=metadata["data_list"][0]
    M=np.asarray(entry["images"]["CAM2"]["lidar2img"],dtype=float)
    assert M.shape==(4,4) and np.isfinite(M).all()
    with zipfile.ZipFile(archive) as z:report=json.loads(z.read("results.json"))
    assert hashlib.sha256(calibration.read_bytes()).hexdigest()==report["source_sha256"]["calibration"]
    w,h=report["image_size"];assert (w,h)==(1242,375)
    for d in report["lidar_detections"]:
        independent=project(d["box3d"],M,w,h)
        assert independent is not None
        assert np.allclose(independent,d["bbox2d"],atol=1e-7,rtol=0)
    assert len(report["lidar_detections"])==18
    return {"status":"PASS","independently_reprojected_3d_boxes":18,
        "calibration_sha256":report["source_sha256"]["calibration"],
        "projection_coordinate_tolerance_px":1e-7,
        "scope":"Original KITTI 000008 calibration consistency, not official KITTI 3D AP"}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("calibration",type=Path);p.add_argument("fusion_archive")
    a=p.parse_args()
    print(json.dumps(verify(a.calibration,a.fusion_archive),indent=2))
