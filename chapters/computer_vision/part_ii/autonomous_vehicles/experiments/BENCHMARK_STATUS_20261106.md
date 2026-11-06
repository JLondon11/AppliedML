# Chapter 5 — empirical benchmark evidence status (2026-11-06)

This file is a repository evidence ledger, **not manuscript prose**. Do not insert
checkpoint hashes, workflow IDs, or production commentary into the printed chapter.

## New genuine CPU inference

### SAM ViT-B on COCO 2017 validation
- Source workflow: https://github.com/JLondon11/AppliedML/actions/runs/41998925403
- Status: **completed successfully**; real pretrained Meta SAM ViT-B, real validation images.
- Protocol: GT box prompts; eight seeded COCO validation images; 23 annotated objects.
- Computed results: SAM mean IoU 0.7799199716256173; mean Dice 0.8554551679333461.
- Matched 15-object subset: SAM mean IoU 0.8277187064118027; GrabCut mean IoU 0.4172375508349529.
- GrabCut comparisons unavailable on eight objects: do not re-code as 0 IoU.
- This is **not** official COCO AP, not a statistically representative benchmark.
- Exact per-instance results and prediction-generated plots are retained in the run artifact.
- Expanded 40-image/120-instance CPU experiment launched: https://github.com/JLondon11/AppliedML/actions/runs/42300131086

### CrowdHuman pedestrian detection
- Original validation archive/ODGT and COCO-pretrained Faster R-CNN ResNet50 FPN workflow:
  https://github.com/JLondon11/AppliedML/actions/runs/42300292512
- 24 seeded original validation images targeted; no numerical predictions may be claimed
  until the workflow succeeds and the evidence artifact is inspected.
- Local score: IoU50 full-body recall stratified by a box-area occlusion proxy.
- These scores must NOT be presented as official CrowdHuman AP, MR, or JI.
- Source photographs are not retained in public GitHub artifacts, per the dataset's
  non-commercial-use and no-redistribution terms.

## Remaining gaps
- PointPillars/KITTI: OpenPCDet's code requires CUDA. A different CPU backend would
  require validated operator parity, checkpoint compatibility, original data, and
  original KITTI evaluation. The existing script fails closed without GPU.
- TuSimple: Original images, a compatible pretrained lane model, the correct
  architecture-specific decoder, and official lane-point metric are needed. Current
  ONNX mask evaluator is not equivalent to published UFLD/SCNN metrics.
- fastMRI: Code presently supports real-data classical zero-fill/TV only; the
  requested neural VarNet evaluation still requires original fastMRI k-space data,
  matching pretrained weights, and model-specific inference.
- NeRF/3DGS: matched real-scene train/test model executions are not available.
  Nerfacto and Splatfacto are not the same models as original Mip-NeRF 360 / 3DGS.
- Camera-LiDAR fusion: requires calibrated paired original sensors, a matching
  trained fusion model, and real held-out detections.
- CrowdHuman/SAM have subset limitations even after successful local runs.

## Acceptance rule
Only publish measured outputs when the actual model ran successfully and an
artifact with per-sample metrics, original data identifier, checkpoint identity,
protocol, and processing code is retained. Keep published-paper benchmark
results explicitly distinguished from independently computed chapter results.
