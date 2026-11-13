# Real Middlebury motorcycle stereo, original Level A experiment

Executed 2026-11-12 with **authentic recorded camera photographs and supplied original reference disparity** through `skimage.data.stereo_motorcycle()`. The actual OpenCV **StereoBM** and **StereoSGBM** algorithms, with identical 96-pixel search limits, were run on the same original rectified stereo pair. This is a **single-scene, offline classical stereo-depth/disparity demonstration**; it is **not** camera–LiDAR fusion, autonomous operation, a neural-method comparison, or a full benchmark.

## Original source and provenance

- Source: scikit-image `data.stereo_motorcycle()`, original Middlebury motorcycle rectified left/right RGB photographs and reference disparity, each **741 × 500**.
- Runtime: Python 3.13.5; scikit-image 0.26.0; OpenCV 4.13.0; NumPy 2.3.5.
- Original source pixel byte SHA-256:
  - left photo: `ca829467c1d4f427da9c4862ba43829da6ac90afe1f75735e95dba9e3fd9620b`
  - right photo: `ae44d83f55e66623c7985499fd2f1685a56023e442e66eca89b3457dd46b17af`
  - reference disparity: `f2c0a477374eb7465e98bca1674c0adb6c536c1c3e05999fb16c68472dc798aa`
- Valid reference mask: finite original disparity >0 and <96 pixels, **343,274** pixels. Error and prediction-coverage computed on valid predicted disparities >0 and <96. All-reference Bad-3 counts invalid predictions as bad, preventing artificially strong results from high rejection.
- No ML fitting/train-validation split was conducted; both deterministic algorithms process the same original photographs and use the external measured disparity as a reference. Split design is inapplicable to this demonstration, not waived to hide a missing model run.

## Executed measured results

| Method | Valid coverage | Valid-pixel MAE | Valid-pixel RMSE | Bad-3 including missing | CPU warm median |
|---|---:|---:|---:|---:|---:|
| OpenCV StereoBM | 74.09% | 1.26717 px | 5.48006 px | 30.40% | 9.09 ms |
| OpenCV StereoSGBM | 82.94% | 1.22261 px | 5.31582 px | 21.88% | 23.87 ms |

These measurements are **specific to this local hardware, software build and single image pair**; they are not general or official Middlebury leaderboard scores. Differences in runtime are environment-dependent. See `original_measurements.csv` for full machine-readable numerical results. Original photographs, ground-truth disparity and model outputs are shown in the local generated `real_middlebury_stereo_comparison.{pdf,png,svg}` scientific figure (downloadable QA release); rerun the committed script to generate them. Do not fabricate images or reuse the numbers for another version of the input.

## How to regenerate

```bash
python -m pip install numpy opencv-python scikit-image matplotlib
python real_middlebury_stereo_certification.py
```

The script writes `original_measurements.csv`, `evidence_provenance.json`, and a scientifically designed six-panel nonbar output figure (original photograph, measured reference disparity, BM and SGBM disparities, spatial error and measured error distributions). The execution is fully reproducible except for normal runtime variations.

## Certification boundary

This **is eligible for a separate Level A certificate**, which requires a scientific assessment of the underlying real-data implementation and figure outputs. It remains **embedded in** the existing Chapter 5 CS06 Multimodal Autonomous Driving case study, which remains scientifically HOLD because no actual camera-LiDAR-fusion model was run. Level A certification cannot be promoted to Level B camera–LiDAR claims, regardless of figure quality. Final Chapter 5 manuscript layout remains a separate production gate.
