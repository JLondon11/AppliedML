"""Local rotated 3D IoU evaluation for genuinely executed PointPillars on KITTI 000008.

Original KITTI reference boxes are camera-coordinate, bottom-center boxes.
Convert via original lidar2cam and yaw-axis transformation. Excludes KITTI
ignore boxes, reports separate 6-all and 4-nonnegative-difficulty Car results.
NOT official KITTI AP, not a hold-out experiment, not a multi-scene benchmark.
"""
import argparse,csv,hashlib,json,math,pickle,zipfile
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolygonPatch

def camera_to_lidar(camera_box,cam2lidar):
    x,y,z,length,height,width,yaw=map(float,camera_box)
    assert min(length,height,width)>0
    # Original KITTI camera boxes are bottom-centered; PointPillars boxes
    # are LiDAR-coordinate, geometric-center boxes.
    xyz=cam2lidar@np.asarray([x,y-height/2,z,1.])
    # Camera local length axis [cos ry,0,-sin ry] maps to LiDAR xy heading.
    facing=cam2lidar[:3,:3]@np.array([math.cos(yaw),0.,-math.sin(yaw)])
    return np.asarray([*xyz[:3],length,width,height,math.atan2(facing[1],facing[0])])

def footprint(box):
    x,y,z,dx,dy,dz,yaw=box
    p=np.array([[-dx/2,-dy/2],[dx/2,-dy/2],[dx/2,dy/2],[-dx/2,dy/2]])
    c,s=math.cos(yaw),math.sin(yaw)
    return p@np.array([[c,s],[-s,c]])+np.array([x,y])

def iou3d(a,b):
    p1,p2=Polygon(footprint(a)),Polygon(footprint(b))
    assert p1.is_valid and p2.is_valid
    overlap_xy=p1.intersection(p2).area
    overlap_z=max(0.,min(a[2]+a[5]/2,b[2]+b[5]/2)-max(a[2]-a[5]/2,b[2]-b[5]/2))
    intersection=overlap_xy*overlap_z
    va=float(a[3]*a[4]*a[5]);vb=float(b[3]*b[4]*b[5])
    return intersection/(va+vb-intersection)

def evaluate(model_artifact,calibration,out):
    with zipfile.ZipFile(model_artifact) as z:pred=json.loads(z.read("metrics.json"))
    # Only load pickle from trusted official upstream OpenMMLab source.
    with calibration.open("rb") as f:scene=pickle.load(f)["data_list"][0]
    lidar2cam=np.asarray(scene["images"]["CAM2"]["lidar2cam"],float)
    assert lidar2cam.shape==(4,4) and np.isfinite(lidar2cam).all()
    cam2lidar=np.linalg.inv(lidar2cam)
    ground_truth=[]
    for label in scene["instances"]:
        if label["bbox_label_3d"]==2:  # KITTI Car class code in source info
            ground_truth.append({"box":camera_to_lidar(label["bbox_3d"],cam2lidar),
                                 "difficulty":int(label["difficulty"])})
    assert len(ground_truth)==6
    assert sum(g["difficulty"]>=0 for g in ground_truth)==4
    proposals=sorted((d for d in pred["detections"] if d["class"]=="Car"),
                     key=lambda x:-x["score"])
    assert len(proposals)==10 and pred["source_points"]==17238
    matrix=np.asarray([[iou3d(np.asarray(d["xyz_dxdydz_yaw"]),g["box"])
                        for g in ground_truth] for d in proposals])
    assert matrix.shape==(10,6) and np.isfinite(matrix).all()
    assert np.all((matrix>=0)&(matrix<=1))
    rows=[]
    for scope,indices in (("all_six_annotated_cars",list(range(6))),
                          ("four_nonnegative_difficulty_cars",
                           [i for i,g in enumerate(ground_truth) if g["difficulty"]>=0])):
        for threshold in (.25,.50,.70):
            matched=set();pairs=[]
            for i in range(10):
                options=[(matrix[i,j],j) for j in indices if j not in matched]
                if options:
                    v,j=max(options)
                    if v>=threshold:
                        matched.add(j);pairs.append({"detector_index":i,"reference_index":j,"iou3d":float(v)})
            rows.append({"scope":scope,"threshold_3d_iou":threshold,
                         "original_reference_cars":len(indices),
                         "pretrained_proposed_cars":10,"matched_reference_cars":len(pairs),
                         "local_reference_recall":len(pairs)/len(indices),
                         "unadjusted_proposal_precision":len(pairs)/10,
                         "matches":pairs})
    out.mkdir(parents=True,exist_ok=True)
    report={"status":"original_KITTI_000008_single_scene_model_vs_original_3D_ground_truth",
            "scope":"one KITTI training scene; local class-Car 3D IoU only, NO official KITTI AP",
            "sample":"000008","original_annotated_cars":len(ground_truth),
            "original_nonnegative_difficulty_cars":4,"actual_pointpillars_car_proposals":10,
            "original_metadata_sha256":hashlib.sha256(calibration.read_bytes()).hexdigest(),
            "original_lidar_sha256":pred["input_sha256"],
            "gt_coordinate_transform":"CAM2 bottom-centered length,height,width; inverse lidar2cam; center and yaw conversion",
            "iou_matrix":matrix.tolist(),"results":rows}
    (out/"measured_3d_iou.json").write_text(json.dumps(report,indent=2)+"\n")
    with (out/"gt_lidar_boxes.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["gt_index","difficulty","x","y","z","dx","dy","dz","yaw"])
        for i,g in enumerate(ground_truth):w.writerow([i,g["difficulty"],*g["box"].tolist()])
    fig,ax=plt.subplots(figsize=(8.3,5.3),layout="constrained")
    for g in ground_truth:
        ax.add_patch(PolygonPatch(footprint(g["box"]),fill=False,
                                  color="#267E91",lw=1.8))
    for d in proposals:
        ax.add_patch(PolygonPatch(footprint(d["xyz_dxdydz_yaw"]),
                                  fill=False,color="#B56F4E",lw=1.0,ls="--"))
    ax.plot([],[],color="#267E91",label="Original KITTI 3D Car annotations")
    ax.plot([],[],color="#B56F4E",ls="--",label="Actual pretrained PointPillars proposals")
    ax.set(xlabel="LiDAR forward x (m)",ylabel="LiDAR lateral y (m)",
           xlim=(0,47),ylim=(-17,11))
    ax.set_aspect("equal");ax.grid(alpha=.12);ax.legend(frameon=False,fontsize=8)
    for ext in ("pdf","png","svg"):
        fig.savefig(out/f"kitti_real_3d_gt_vs_pointpillars.{ext}",dpi=210,bbox_inches="tight")
    for row in rows:
        print("MEASURED_3D_GT",row["scope"],row["threshold_3d_iou"],
              row["matched_reference_cars"],row["original_reference_cars"],flush=True)
    print("COMPLETED_ORIGINAL_GT_SINGLE_SCENE",json.dumps({k:v for k,v in report.items() if k not in ("results","iou_matrix")}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model_artifact");p.add_argument("original_kitti_metadata",type=Path)
    p.add_argument("--out",type=Path,default=Path("kitti_3d_gt_local_results"))
    a=p.parse_args();evaluate(a.model_artifact,a.original_kitti_metadata,a.out)
