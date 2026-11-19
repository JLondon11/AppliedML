# Chapter 5 — original-model scientific certification and real-image print remediation

**Scientific pass date:** 2026-11-18 (Arizona); external GitHub Actions events appear as 2026-11-19 UTC. **Governing scientific standard:** repository-root proportional original-data R1–R10, with distinct scientific and manuscript-production decisions. **Final decision:** 4 explicitly scoped certified experiments, including exactly 1/11 full parent applications, with 3 separate embedded Level A demonstrations. **Chapter 5 publication release: HOLD** until actual current compiled manuscript and all original model/application evidence are independently approved.

## Current scientific decision — do not confuse parent studies and demonstrations

- **CV5-CS09, Level B scientific CERTIFIED:** original COCO 42-scene box-prompted frozen SAM ViT-B instance segmentation and 147 mask results, including 99 paired same-box GrabCut comparison samples. This does not assert prompt-free universal instance segmentation, on-road deployment or the latest full PDF figure integration.
- **CV5-D01, Level A scientific CERTIFIED:** original frozen YOLO11n inference on 24 selected VisDrone photographed validation *files* with **23 unique source image byte hashes** and inconsistent distinct annotation identities on one exact repeated photo. Original local AP50 **0.140645633**, not official VisDrone AP. No video tracking, drone planning or flight control; CV5-CS03 parent remains HOLD.
- **CV5-D02, Level A scientific CERTIFIED:** original frozen Faster R-CNN inference on 170 distinct Penn–Fudan photographs, 423 ground-truth people, 1,199 retained predictions and 417 matching true positives; local recorded AP50 **0.968183484**, not claimed independently recomputed without masks. No occlusion-stratified annotation/evaluation; CV5-CS11 parent remains HOLD.
- **CV5-D03, Level A scientific CERTIFIED:** genuinely executed original Middlebury motorcycle photographic stereo pair, actual StereoBM and StereoSGBM source algorithms with original disparity truth. This *never* certifies the CV5-CS06 **camera–LiDAR fusion** parent, which requires original synchronized camera and LiDAR inputs and fused inference.
- **Remaining full parent applications:** 10/11 scientific HOLD, not silently downscoped by 3 valid narrow experiments. Level C original parents CV5-CS04/CV5-CS05 remain Level C; all other original parent units remain Level B.

Source registries:
`qa/CH5_SCIENTIFIC_VS_PRODUCTION_CERTIFICATION_20261112.csv`;
`qa/CH5_LEVEL_A_EMBEDDED_DEMONSTRATION_CERTIFICATIONS_20261112.csv`.

New independent cross-register gate: `qa/validate_current_ch5_certificate_accounting.py`, used in `.github/workflows/chapter5_science_vs_production_gate.yml` and original detector-archive CI. This guards the **11+3 separate records, required parent links, original 24 versus 23 VisDrone file/content count, exact ten science requirements, and distinct production HOLD statuses**. Its GitHub Actions science-versus-production run **44430562685 completed SUCCESS**.

## Actual new original-data reproduction, not an invented illustration

Original archived Middlebury source `skimage.data.stereo_motorcycle()` provided true **500×741 left/right photographs** and original reference disparity. The actual original OpenCV **StereoBM** and **StereoSGBM** implementations were executed again locally with the previously certified disparity cap 96 and source parameters. The original disparity reference has **343,274 valid pixels**; no train/validation/test split is invented for these two fixed classical methods.

| Measured original Middlebury result | StereoBM | StereoSGBM |
|---|---:|---:|
| Reference-valid pixels | 343,274 | 343,274 |
| Original predicted-valid pixels | 254,330 | 284,723 |
| Coverage of reference | **74.0895%** | **82.9434%** |
| Valid-prediction mean absolute disparity error | 1.26717 px | 1.22261 px |
| Valid-prediction root mean square disparity error | 5.48006 px | 5.31582 px |
| Bad-3 counting reference-valid missing predictions as failures | 30.4008% | 21.8764% |
| This rerun's warm CPU median | 6.40 ms | 22.62 ms |

The camera results are genuine measured source-model outputs; they are not synthesized labels, learned neural depth predictions or evidence of LiDAR. The warm CPU latencies depend on original hardware and do **not** have to match the archived published workstation's medians; the numerical disparity/error/coverage values matched to the original measurement precision.

### New professional scientific figure presentation: two image-first print-width plates

The originally published scientific comparison source had a single **11.8-inch-wide** six-panel PDF. At normal 6.5-inch textbook insertion it risks tiny captions, insufficient source-image resolution and excess plotting density. A newly executed source-fidelity figure generator was **committed** at:
`figure_generation/real_middlebury_stereo/compact_print_stereo.py`.

It reuses the chapter's actual original classical inference implementation (no substituted model), and produces **two scientific original-data image plates**:

1. `middlebury_stereo_source_and_actual_predictions.pdf` — authentic left and right photographic input, actual provided original disparity ground truth, actual BM and SGBM disparity images, and original photograph close-up.
2. `middlebury_stereo_observed_error_and_missing.pdf` — genuine BM/SGBM absolute-disparity-error maps masked by both original GT validity and prediction validity; independently visualized pixel coverage masks and original-source close-ups to identify exactly where predictions are missing or inaccurate.

**A photograph/model-result gallery is a more scientifically useful image than another wide bar chart.** No fabricated source photos or derived ground-truth images are substituted. The figures use shared scale 0–90 disparity pixels and 0–12 absolute error pixels, disclose the single-scene scope and name missing predictions explicitly.

The original algorithm-and-new-figure workflow `.github/workflows/chapter5_real_middlebury_stereo_level_a.yml` successfully re-executed as **GitHub Actions run 44430400388**, preserving an original-rerun artifact **14485429821**. The binary archive was downloaded in this scientific pass and independently inspected: both **468-point = 6.5-inch-wide** original-model source figure PDFs have **6 actual image panels, minimum selectable figure text 8.2 pt, zero off-page text**, and a complete PDF/PNG/SVG output triple. The archived original image metrics exactly match independently recomputed original model results; measured per-run clock timings differ across execution environments as expected. The downloaded artifact passed ZIP member/CRC validation.

**Observed CI PDF SHA256:** source/prediction plate `7754a97aeb39b20d7a8b11dfe3759ccd8c2b0bf0ab7467a1cf84af74c3dbd1b8`; error/missing plate `85d2c8c945040576cdd46bd5d53edd83f320db75e0de2832ad67611d8aba905f`. PDF container creation timestamps may affect serialized byte hashes between independently generated runs; actual source/model numbers and intended printed content remain the scientific invariants.

## Scientific R1–R10 conclusion

CV5-D03's prior proportional **ten R1–R10 PASS** judgments remain valid for the **single-scene classical stereo algorithm demonstration**. This pass strengthens R3/R4 actual rerun, R6 original mask-aware error and coverage reporting, R8 two genuine source/model image-first plates, and R9 original executable code, machine-observed outputs and CI source provenance. No new independent source scene is invented, no scene-level confidence intervals are asserted and no parent application is promoted without task-matched evidence.

Remaining original Chapter 5 tasks, in priority order:

- **CS06 true camera–LiDAR fusion:** load genuine synchronized calibrated original image/point-cloud scenes and execute a real fusion algorithm with matched single-sensor references; cannot extrapolate from stereo.
- **CS11 true occlusion-stratified pedestrian evaluation:** genuine original occlusion labels and matched actual model predictions; Penn–Fudan general pedestrian detection alone is not occlusion evidence.
- **CS03 autonomous drone:** original chronological multi-frame identities/association metric and only actually supported planning/flight-control tasks, or retitle the parent manuscript study to its proven still-image task with separate explicit certification.
- **CS01, CS07, CS08, CS10:** task-matched original KITTI 3D, fastMRI, NeRF/3DGS, and TuSimple models/datasets and real outputs; code-only preflights are not experiment certificates.
- **CS02, CS04, CS05:** genuine agricultural-field, cross-dataset driving and real production-stack experiments where the original titles require them.

## Production decision

Both newly generated stereo plates are **scientifically verified and print-width staged**, not automatically assigned new Chapter 5 figure numbers or inserted in a latest compiled manuscript. The current complete Chapter 5 LaTeX/PDF has not been independently retrieved, and no 11-parent full-production figure/caption/table/code audit has been completed. Figure panel images/pixel evidence are in the original real-data Actions artifact, and the complete new Python source is committed to its corresponding GitHub section. **Final Chapter 5 Master QA production status: HOLD.**
