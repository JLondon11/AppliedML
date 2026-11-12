# Chapter 5: outstanding science continuation — 2026-11-11

**Master QA: HOLD; no new TuSimple or CrowdHuman model inference has been executed.**

## Official source verification

- TuSimple official guideline: https://github.com/TuSimple/tusimple-benchmark/blob/master/doc/lane_detection/guideline.md ; annotated lane x coordinates are specified at original h_samples y coordinates, and the task's official evaluation is lane-point based. Local 4-pixel rasterized polylines / pixel IoU from the existing ONNX evaluator are not leaderboard scores.
- CrowdHuman original documentation: https://www.crowdhuman.org/download.html ; ODGT annotations contain full-body, visible-body and head boxes plus ignore/occlusion information. Local occlusion-area-proxy stratification is not an official AP/MR/JI protocol.

## This pass: source-level scientific enhancements committed

1. `case_studies/occluded_pedestrian_detection/evaluate_crowdhuman_frcnn.py` now saves **every retained original-model person box and confidence score** and **SHA-256 of every original JPEG**, along with the already available per-person matching and aggregate recall CSV. Commit `18424a530de6f37b9ccb436669f1922d870474d2`.
2. New `case_studies/occluded_pedestrian_detection/plot_crowdhuman_occlusion.py` produces restrained-color measured-only original person recall figures with original-photograph-cluster bootstrap intervals (5,000 replicates by default), and writes the plotted numerical values and input CSV SHA-256. The script rejects missing/empty measurement records and requires observed people in all displayed bins. Commit `fbe8e784e921a78bdd7de5fe3024b98e5fedaa10`.
3. The corresponding README documents the post-inference figure command and its statistical scope. Commit `d3b2d7636d51280630b625d1f9168f872509c1ee`.
4. Previous TuSimple semantics corrections are retained: `198cde84a40a3a636e0517cc78c2a7122af51b2e`, `0fd61010e67465ad997cfaeb9048d161dc2ac568`, and `e82bf0c4b69ec69a3c0e83554f0c98d6c798840c`.

## Execution checks and exact blockers

- The available local runtime has CPU PyTorch, torchvision, NumPy, OpenCV and Matplotlib. A locally authored compatible CrowdHuman plot script was syntax-checked, its CLI help was executed, and its missing-input failure was observed. These are **software checks only**, not model inference or complete suite certification.
- Original CrowdHuman JPEGs, validation ODGT and exact pretrained Faster R-CNN checkpoint are not present in the current local runtime.
- Original TuSimple video/image data and compatible lane-specific ONNX weights are not present in the current local runtime. ONNX Runtime also is not installed locally. Public official TuSimple download is ~9 GB for one documented package, and CrowdHuman has official external dataset links; neither is materialized here.
- No GitHub Actions dispatch API was available in the connected GitHub toolset for starting a new GPU run. Existing successful model runs on **other datasets** do not satisfy either named-task experiment.
- The exact current Chapter 5 TeX/PDF binary manuscript is not locally available, therefore no new Chapter 5 figure was inserted or publication-certified in this pass.

## Next factual promotion gates

- **CS10 TuSimple:** obtain original dataset, named trained lane checkpoint, explicit preprocessing/output semantics, execute real model, save raw predictions, validate official lane-level metric independently, generate overlay and failure figures, integrate with revised text and render PDF.
- **CS11 CrowdHuman:** obtain original evaluation JPEGs/ODGT and exact COCO Faster R-CNN weights, run the committed evaluator, retain full prediction/image hash CSV, generate its measured-only uncertainty figure, execute official metric protocol for official AP/MR claims, examine heavily occluded failure examples, integrate into PDF.
- Then **KITTI PointPillars**, **camera–LiDAR fusion**, and other 7 remaining units per `CH5_CODE_MANIFEST.csv`.

**No new scientific score or original-dataset plot is claimed. No unit is promoted to full certification.**
