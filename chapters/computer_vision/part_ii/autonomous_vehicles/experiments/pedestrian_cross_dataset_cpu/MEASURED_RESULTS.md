# Executed same-model pedestrian transfer on original data

**Run:** https://github.com/JLondon11/AppliedML/actions/runs/42431710322

**Actual model:** torchvision COCO-pretrained Faster R-CNN ResNet50-FPN (person class), CPU.  
**Inputs:** original PennFudanPed photographs and instance-mask annotations; original CrowdHuman validation photographs and full-body bounding-box annotations.  
**Protocol:** 24 original photographs per dataset; confidence >= 0.30; person class; one-to-one full-body IoU >= 0.50; unchanged model weights and inference settings.  
**Sampling:** first 24 sorted Penn-Fudan image filenames; 24 CrowdHuman validation images selected by seed 20261106.

| Dataset | Images | Reference persons | Matched persons | Detections | Local recall |
|---|---:|---:|---:|---:|---:|
| Penn-Fudan | 24 | 39 | 39 | 112 | 1.0000 |
| CrowdHuman | 24 | 476 | 259 | 555 | 0.5441176471 |

**Evidence artifact:** `chapter5-pennfudan-crowdhuman-matched-detector-evidence` contains `per_image.csv`, `metrics.json`, and genuine code-generated PNG/PDF/SVG. Inputs and pretrained weights were downloaded in GitHub Actions and not redistributed.

## Scientific qualification

This comparison measures person recall **only**, not precision or AP. The 112 and 555 detections include false positives, duplicates and mismatches. Penn-Fudan is a small and much less crowded dataset than CrowdHuman; their sampling and original annotation conventions differ. The difference in local recall is descriptive and cannot isolate dataset shift, occlusion, or model architecture causally. Neither subset consists of KITTI/Waymo/nuScenes/Argoverse vehicle driving frames, so this experiment **does not certify** the Chapter 5 cross-dataset autonomous-driving 3D benchmark. It must not be compared numerically to published vehicle-detection mAP results.
