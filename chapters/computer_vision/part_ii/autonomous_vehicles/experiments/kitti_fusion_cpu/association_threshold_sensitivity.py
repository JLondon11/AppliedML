"""Actual KITTI camera/LiDAR detection association threshold sweep.

Computed from archived model predictions only. Measures box-association
sensitivity, never official KITTI object detection AP or accuracy.
"""
import argparse,json,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def iou(a,b):
    x=max(a[0],b[0]);y=max(a[1],b[1])
    xx=min(a[2],b[2]);yy=min(a[3],b[3])
    inter=max(0,xx-x)*max(0,yy-y)
    aa=(a[2]-a[0])*(a[3]-a[1]);bb=(b[2]-b[0])*(b[3]-b[1])
    return inter/(aa+bb-inter) if aa+bb-inter>0 else 0

def calculate(report):
    assert len(report["lidar_detections"])==18 and len(report["camera_detections"])==15
    mapping={"Car":"car","Pedestrian":"person","Cyclist":"bicycle"}
    possible=[]
    for i,l in enumerate(report["lidar_detections"]):
        for j,c in enumerate(report["camera_detections"]):
            if mapping[l["class"]]==c["class"]:
                possible.append((iou(l["bbox2d"],c["bbox2d"]),i,j))
    possible.sort(reverse=True)
    out=[]
    for threshold in (.10,.20,.30,.40,.50,.60,.70,.80,.90):
        seenl,seenc=set(),set();matches=[]
        for v,i,j in possible:
            if v<threshold:break
            if i not in seenl and j not in seenc:
                seenl.add(i);seenc.add(j)
                matches.append((i,j,v))
        if threshold==.20:
            assert len(matches)==9
            for expected,actual in zip(report["matches"],matches):
                assert (expected["lidar_index"],expected["camera_index"])==actual[:2]
                assert abs(expected["projected_box_iou"]-actual[2])<1e-12
        out.append({"minimum_projected_IoU":threshold,
                    "matched_pairs":len(matches),
                    "LiDAR_box_match_fraction":len(matches)/18,
                    "camera_box_match_fraction":len(matches)/15})
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("model_artifact_zip")
    p.add_argument("--out",default="kitti_sensitivity")
    a=p.parse_args()
    with zipfile.ZipFile(a.model_artifact_zip) as z:
        report=json.loads(z.read("results.json"))
    results=calculate(report)
    output=Path(a.out);output.mkdir(parents=True,exist_ok=True)
    (output/"association_threshold_sensitivity.json").write_text(json.dumps({
        "scope":"Original KITTI 000008 predicted box association sensitivity; NOT accuracy",
        "threshold_sweep":results},indent=2)+"\n")
    fig,ax=plt.subplots(figsize=(7.3,3.8),layout="constrained")
    ax.plot([r["minimum_projected_IoU"] for r in results],
            [r["matched_pairs"] for r in results],
            marker="o",lw=1.7,color="#287D91")
    ax.scatter([.20],[9],s=75,color="#B56F4E",zorder=3,
               label="Measured study threshold = 0.20")
    ax.set(xlabel="Minimum projected 2D cross-sensor IoU",
           ylabel="Greedy one-to-one matched detections",
           xlim=(.07,.93),ylim=(0,13))
    ax.set_yticks(range(0,14,2));ax.legend(frameon=False,fontsize=9)
    ax.grid(alpha=.16)
    for ext in ("pdf","png","svg"):
        fig.savefig(output/f"kitti_association_threshold_sensitivity.{ext}",
                    dpi=210,bbox_inches="tight")
    print(json.dumps(results,indent=2))

if __name__=="__main__":main()
