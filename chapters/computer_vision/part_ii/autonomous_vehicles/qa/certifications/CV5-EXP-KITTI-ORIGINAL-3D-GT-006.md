# CV5-EXP-KITTI-ORIGINAL-3D-GT-006 — CERTIFIED (bounded original KITTI 3D localization experiment)

**Date:** 2026-11-08. **Approval level:** one original annotated KITTI detection **training** scene, genuine pretrained PointPillars 3D LiDAR inference and actual calibrated 3D ground-truth comparison. This is NOT full KITTI 3D AP, NOT held-out object-detection generalization, and NOT an end-to-end real-time benchmark.

## Exact executed experiment and provenance

- Original source scene: KITTI detection scene **000008**, MMDetection3D's original KITTI LiDAR demo mirror; 17,238 measured XYZI returns.
- Original point-cloud SHA256 `3b9de6cc966534900f6a1bdc93b21772e47a334eb2ef18082021956520d902d1`.
- Original KITTI camera-LiDAR metadata SHA256 `b187a5f9185789125e04618645ecfb7c5e9afb5d41ab11b06b9a0cf381f0c45a`.
- Executed actual pretrained `torch-pointcloud` OpenPCDet-compatible PointPillars `pointpillars.kitti.openpcdet`, class-aware rotated NMS 0.01 and detection confidence >0.20, CPU model inference. The actual decoded network produced 18 oriented 3D boxes in total, including 10 Car proposals.
- Genuine model execution run https://github.com/JLondon11/AppliedML/actions/runs/42354827106.
- Original annotated 3D Car-box and actual prediction evaluation https://github.com/JLondon11/AppliedML/actions/runs/42531178943.
- Downloaded original 3D evidence artifact ZIP SHA256 `8280702295c1b65a2c10bb17f9812c50e29c3276fe5e6086e9f37ee198c83d51`.
- Fail-closed mathematical and result-consistency verification https://github.com/JLondon11/AppliedML/actions/runs/42531928712.

## Model-to-original-ground-truth spatial protocol

Original KITTI camera-coordinate bottom-centered 3D car annotations are transformed to geometric centers by subtracting half camera-box height from camera y and then applying the inverse original `lidar2cam` calibration. Camera local forward-axis orientation is transformed into LiDAR yaw. Oriented LiDAR 3D boxes are compared using **actual polygon footprint overlap times z-axis overlap**, not axis-aligned 2D proxy boxes. Car predictions are greedily matched once to each reference in score order.

**Verified outcomes:** Original scene contains six 3D annotated Car objects; four have a nonnegative official KITTI difficulty code. Among ten actual model Car proposals, six distinct references are matched at each local 3D IoU cutoff 0.25, 0.50 and 0.70; all four nonnegative-difficulty references match at 0.70. The six uniquely matched overlaps are approximately 0.907, 0.894, 0.885, 0.745, 0.748 and 0.839. **These are single-frame observations; they must not be described as a 100% KITTI recall estimate.**

## Ten scientific criteria

1. **PASS:** original KITTI annotated 3D detection scene directly aligns with genuine PointPillars Car localization.
2. **PASS for single-scene evidence:** authentic original point cloud, calibration and GT; explicitly training scene, **not** an independently held-out validation benchmark.
3. **PASS:** original named pretrained PointPillars model executed and raw decoded 3D predictions archived.
4. **PASS:** actual 17,238-point scene; 18 retained boxes/10 Cars; original annotation population; confidence and NMS, CPU model inference configuration; no local model training.
5. **PASS at declared single-scene scope:** genuine rotated 3D IoU computed for original boxes and threshold sensitivity at 0.25/0.50/0.70, with annotated ground truth as direct reference. No official AP or multi-frame ranking.
6. **PASS:** actual original LiDAR BEV/prediction figure and original 3D GT-versus-network prediction figure from executed code; new Table 5.3 and Figure 5.29 integrated with scientific discussion.
7. **PASS:** complete PointPillars runner, calibrated 3D GT evaluator, original source data and hash, measured output artifact, new fail-closed verification script and GitHub Actions workflows in section-specific repo directories.
8. **PASS:** locally measured single-scene 3D IoU separated explicitly from the published PointPillars 16.2 ms stage-timing numbers and official KITTI AP.
9. **PASS at single-scene scope:** IoU threshold sensitivity, duplicate proposal/GT count checks, source selection and geometry limitations discussed; statistically representative between-scene confidence interval **not applicable** to a one-scene illustration.
10. **PASS:** independent mathematical overlap sanity tests (symmetry, exact self match, disjointness and angle periodicity), one-to-one match/overlap matrix reconciliation, duplicate prevention, recompiled 125-page Chapter 5 PDF and visual inspection of the ground-truth result page, no unresolved references or overfull boxes.

**Limitations and non-certified parent unit:** This evidence cannot establish full KITTI class/difficulty 3D AP40, across-frame recall, a held-out generalization claim, repeated latency or an automotive real-time bound. The parent **KITTI 3D Object Detection Benchmarking and Real-Time Inference Latency** case study remains **PENDING**, because those named benchmark and end-to-end timing requirements need separate genuine executions.

**Review classification:** internally verified and scientifically bounded graduate-level original-data experimental certificate, not third-party accreditation.
