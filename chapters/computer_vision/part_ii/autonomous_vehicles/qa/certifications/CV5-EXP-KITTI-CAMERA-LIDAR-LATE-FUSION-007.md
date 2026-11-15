# CV5-EXP-KITTI-CAMERA-LIDAR-LATE-FUSION-007 — CERTIFIED (bounded genuine two-model late fusion)

**Date:** 2026-11-08. **Scope:** geometric, class-aware camera/LiDAR **decision-level association**, not a jointly trained feature-level fusion network, and not a complete multimodal/autonomous-driving benchmark.

## Original real-data model execution and independently verified outcomes

- Actual original synchronized KITTI 3D detection **training** scene `000008`, original camera 2 frame of 1242×375 pixels and original LiDAR XYZI cloud with **17,238 points**.
- Original original-data source: upstream OpenMMLab MMDetection3D KITTI scene and original calibrated LiDAR-to-CAM2 projection. Original LiDAR SHA256: `3b9de6cc966534900f6a1bdc93b21772e47a334eb2ef18082021956520d902d1`; original camera SHA256: `5b988d2a04d51850610b38ce50a66fd4027f3f5e645e5f2198d0522f4cf9a640`; original calibration SHA256: `b187a5f9185789125e04618645ecfb7c5e9afb5d41ab11b06b9a0cf381f0c45a`.
- Both **genuine** pretrained networks executed on the same original synchronized scene: OpenPCDet-compatible PointPillars 3D LiDAR detection and torchvision COCO-pretrained Faster R-CNN ResNet50-FPN RGB-camera detection. Execution workflow: https://github.com/JLondon11/AppliedML/actions/runs/42356021439 .
- Real model box counts: **18** LiDAR 3D boxes projected into camera 2 (LiDAR score ≥0.20); **15** original-camera detections (camera score ≥0.25); **9** uniquely matched class-consistent projected-box associations, at 2D IoU ≥0.20.
- Actual source evidence archive comprises `results.json`, `matches.csv`, and code-generated figure `kitti_calibrated_late_fusion.{pdf,png,svg}`. The original KITTI photographs are not bundled into public model-output artifacts.
- The independently written `verify_original_kitti_late_fusion.py` recomputes every matched projected-box IoU, source hash constraints, score and box bounds, class mapping and greedy uniqueness. Independent CI: https://github.com/JLondon11/AppliedML/actions/runs/42532901004 .
- The original chapter includes this measured experiment's figure and narrative in the multimodal perception section. The original model-only CPU timing measurements are not interpreted as end-to-end latency.

## Ten scientific criteria: bounded experiment

1. **PASS**: Actual same-scene image/point-cloud camera–LiDAR object association is precisely matched to the named two-model decision-level fusion task.
2. **PASS in original-scene scope**: original synchronized KITTI camera, LiDAR and source calibration; single **training** scene explicitly excluded from held-out performance claims.
3. **PASS**: genuine pretrained PointPillars and Faster R-CNN inference on original images and points; no mocked boxes.
4. **PASS**: scene size, detector configurations and thresholds, fixed calibration geometry, scores and modeled training status documented.
5. **PASS for association demonstration**: independently computed projected 2D IoU and unique class-consistent matches; no official 3D AP or retrieval accuracy claim.
6. **PASS**: real same-scene camera detections, LiDAR projected boxes and nine associations rendered into a code-generated figure and discussed in the multimodal section.
7. **PASS**: complete section-organized original model code, source data SHA256s, recorded outputs, visualization and independent audit script committed to GitHub.
8. **PASS**: measured results separated from published benchmarks and from unexecuted jointly trained BEVFusion, MVX-Net and TransFusion.
9. **PASS proportionate to deterministic single-scene demonstration**: unmatched detections, threshold choice, projection assumptions, lack of independent ground-truth association IDs and limits disclosed; population uncertainty not meaningful on one scene.
10. **PASS**: independently rederived all nine matches, original scene constraints and score/2D box geometry, original result figure integrated with text. CI audit and code compile-check both succeeded.

## Noncertified claims and parent unit status

The nine associations do **not** measure ground-truth tracking/re-identification accuracy, 3D detection improvement from combining sensors, a jointly trained early/feature fusion architecture, official KITTI/nuScenes/Waymo AP, robustness across weather/scenes, repeated inference throughput, or autonomous-vehicle deployment. Without these, the parent `Multimodal Autonomous Driving Pipeline` case study remains **PENDING**. The separate `KITTI 3D Object Detection Benchmarking and Real-Time Inference Latency` case study also remains pending.

**Decision:** internally verified **bounded experiment certified**; no third-party accreditation or full-case-study approval implied.
