"""Determine whether the original KITTI demo scene includes GT 3D labels.

Prints only metadata schema, class counts, and usable ground truth existence.
Inspection is NOT an object-detection benchmark.
"""
import argparse,pickle,json
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument("metadata",type=Path);a=p.parse_args()
    # Only use original trusted official OpenMMLab source: unpickling arbitrary sources unsafe.
    with a.metadata.open("rb") as f:data=pickle.load(f)
    print("ROOT_KEYS",sorted(str(x) for x in data.keys()))
    print("META_KEYS",sorted(str(x) for x in data.get("metainfo",{}).keys()))
    records=data.get("data_list",[])
    print("SCENES",len(records))
    for scene in records[:1]:
        print("SCENE_KEYS",sorted(str(x) for x in scene.keys()))
        print("CAM2_KEYS",sorted(str(x) for x in scene["images"]["CAM2"].keys()))
        for key in ("lidar2cam","cam2img"):
            value=scene["images"]["CAM2"].get(key)
            print("CALIBRATION",key,value)
        print("INSTANCE_COUNT",len(scene.get("instances",[])))
        for k,x in enumerate(scene.get("instances",[])):
            print("GT_3D_INSTANCE",k,"CLASS",x.get("bbox_label_3d"),"BOX",x.get("bbox_3d"),"DIFFICULTY",x.get("difficulty"))
        for x in scene.get("instances",[])[:4]:
            print("INSTANCE_KEYS",sorted(str(y) for y in x.keys()))
            print("INSTANCE_CLASS",x.get("bbox_label_3d",x.get("name","unknown")))
        print("ANN_INFO_KEYS",sorted(str(x) for x in scene.get("ann_info",{}).keys()))
        print("KITTI_GT_AVAILABLE",bool(scene.get("instances") or scene.get("ann_info")))
if __name__=="__main__":main()
