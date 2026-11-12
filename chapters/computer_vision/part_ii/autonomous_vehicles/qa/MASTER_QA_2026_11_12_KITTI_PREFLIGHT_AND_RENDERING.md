# Chapter 5 — KITTI PointPillars scientific execution prerequisite review (2026-11-12)

**Chapter:** Computer Vision Part II (Chapter 5). **Overall scientific certification:** HOLD. This is a *source and preflight validation remediation*, not a claim of actual KITTI PointPillars model execution.

## Why this was the next priority

The repository already contains an OpenPCDet PointPillars runner, genuine-LiDAR renderer and geometry tests. The named experimental inputs—official KITTI 3D benchmark LiDAR, labels/calibration and matching trained PointPillars checkpoint—are not present in the current local workspace. Their absence prevents actual inference, official AP, and any new real-model KITTI result figure.

## Scientifically consequential code corrections

1. **Partial source validation eliminated.** Previous `validate_inputs` checked that scans existed for only the *first five* IDs in `ImageSets/val.txt`. Now every original validation ID is checked for a six-digit unique frame ID, corresponding original nonempty LiDAR XYZI scan, matching calibration and label path, and a LiDAR file whose byte length is divisible by 16. Each validation split's SHA-256 is recorded in the run manifest. Commit `d3658d1b9d8c5adc1561ea2ca89aafae7a6c2e19`.
2. **Empty negative-scene labels are allowed.** A valid zero-object KITTI frame can have an empty annotation text file. The strengthened preflight validates existence but does not reject an empty label. Commit `c4de426ac3068bd95f8a4338395dd0497895e0f1`.
3. **Prediction validation strengthened.** The renderer refuses zero/negative 3D box extents, NaN/Inf or out-of-range model confidence scores, and duplicated displayed frame IDs. Commit `aed48f5f8c1f5d1d61428d99b19a282004ed4708`.
4. **Preflight regression cases committed.** `case_studies/kitti_3d_detection/test_local.py` now includes six frame fixtures (to confirm late-ID detection), missing-scan rejection, malformed/duplicate IDs, empty zero-object label acceptance, invalid LiDAR record length, and erroneous box/score rejection. These are **code unit-test fixtures only**, never manuscript data. Commits `66a4a230c154146128c7ad3bb43acdbc6c8f7a87`, `ce98462bebaac12631d30946443f265c52976092`, `cc5219e28eb11e11b6696bec9da00d0a9b142ae0`, `e49c3fbc50730a651abea2703b895ae22ba6db4d`.
5. **CI validation workflow:** `.github/workflows/chapter5_kitti_preflight_integrity.yml` executes syntax checks and the focused Python test on repository changes (commit `821e248aec7c08bf8183a37179fc96c99f1eea63`). It is not a benchmark, and successful remote GitHub Actions execution has **not yet been independently verified**.

## Verified vs unverified evidence

Verified from directly reading connected GitHub files: concrete original source-code corrections and successful source commits. Original KITTI files and PointPillars checkpoint were not present in `/mnt/data` in this session; no model execution and no original-dataset AP were performed. Local fixture acceptance and CI success on the *newly committed changes* have **not** been established; the tests are committed for execution.

A successful run of `test_local.py` will establish preflight/geometry logic only—not validity of the trained model, official OpenPCDet evaluation or visual result assets.

## Required next run

- Obtain official KITTI validation split/images, point clouds, calibration files and ground-truth annotations, and a published compatible OpenPCDet PointPillars config/checkpoint.
- Pin OpenPCDet and CUDA/PyTorch versions, source/split/weight hashes; execute complete validation split with `run_pointpillars_kitti.py`.
- Preserve original `result.pkl`, evaluation logs and official 3D AP by class/difficulty, plus latency with device and batch details.
- Generate actual predicted 3D boxes on original LiDAR scans with `render_verified_pointpillars.py`, compare to original annotations, and visually inspect each generated figure/caption.
- Integrate in current Chapter 5 TeX, rebuild and inspect PDF. No new figures have been inserted because the current Chapter 5 LaTeX/PDF release was not supplied in the local workspace.

**Scientific decision:** KITTI evaluation preflight and output-geometry guards strengthened; CV5-CS01 and overall Chapter 5 remain NOT CERTIFIED.
