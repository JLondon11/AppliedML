# Genuine KITTI original 3D ground truth versus executed PointPillars

## Original-data executions

- Pretrained OpenPCDet-compatible PointPillars on the original KITTI 000008 XYZI scene: [Actions 42354827106](https://github.com/JLondon11/AppliedML/actions/runs/42354827106).
- Original KITTI 3D annotated ground truth, calibrated 3D axis conversion, rotated box 3D IoU against the actual original network output: [Actions 42531178943](https://github.com/JLondon11/AppliedML/actions/runs/42531178943).
- Fail-closed independent 3D intersection, axis transform / match consistency test: [Actions 42531928712](https://github.com/JLondon11/AppliedML/actions/runs/42531928712).

## Reproduce

Run `execute.py` using the environment and command in `.github/workflows/chapter5-pointpillars-kitti-cpu.yml`; retain the original `metrics.json`. Fetch upstream MMDetection3D `demo/data/kitti/000008.pkl` from the trusted OpenMMLab repository, and call:

```sh
python evaluate_single_scene_original_3d_gt.py pointpillars_run_artifact.zip 000008.pkl --out scored_results
zip -q -j scored_results.zip scored_results/*
python verify_original_3d_gt.py scored_results.zip
```

The evaluator requires the actual `metrics.json` stored at the root of the ZIP. Original point clouds and images are **not** redistributed in workflow artifacts.

## Measured scene-specific evidence

- Original scene `000008`, 17,238 LiDAR XYZ-intensity returns, 18 decoded retained detections, ten class-Car oriented 3D proposals.
- Six original KITTI 3D Car annotations; four have nonnegative KITTI difficulty.
- Genuine oriented 3D overlap at IoU thresholds 0.25, 0.50, 0.70 matches six of six original cars; all four eligible difficulty-coded cars also match at 0.70.
- Exact single-scene predicted/reference maximum overlaps: 0.907, 0.894, 0.885, 0.745, 0.748, 0.839.
- Original source hashes and detector configuration in `measured_3d_iou.json` and run `metrics.json`.
- The matching plot `kitti_real_3d_gt_vs_pointpillars.{pdf,png,svg}` is generated from the actual reference and predicted 7-parameter 3D boxes, not a synthetic reconstruction.

## Scientific restrictions

This experiment executes **one original KITTI training scene** and only Car objects. It is **not** official KITTI AP40, a validation benchmark, a confidence interval across scenes, a test of pedestrian/cyclist performance, a second model comparison or repeated vehicle end-to-end inference latency. The 0.506 s recorded CPU model/decoder latency is not comparable to the original PointPillars paper's published GPU 16.2 ms stage-total.

**Bounded evidence decision:** `qa/certifications/CV5-EXP-KITTI-ORIGINAL-3D-GT-006.md` certifies the executed 3D GT experiment only. The complete KITTI benchmarking-and-latency case-study remains **PENDING**.
