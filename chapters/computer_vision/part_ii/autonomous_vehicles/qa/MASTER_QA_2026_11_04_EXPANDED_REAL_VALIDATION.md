# Chapter 5 — expanded genuine pretrained-model validation (2026-11-04)

**Chapter:** 5, Computer Vision Part II: Autonomous Systems. **Chapter-wide Master QA: NOT APPROVED**.

## Verifiable successful executions

1. **SAM ViT-B (original checkpoint, original COCO real segmentation masks)**: https://github.com/JLondon11/AppliedML/actions/runs/41980537230. Expanded independently from earlier three-image/eight-object pilot to 18 distinct fixed-seed randomized, eligible COCO validation photographs with **59 original instance masks** and **35 valid matched box-prompt GrabCut comparisons**. Checkpoint SHA256 `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`. No fine-tuning; ground-truth boxes are oracle prompts. All object-level metrics, original-image SHA256 values, and actual prediction figures in original run artifact; summary numeric evidence committed under `case_studies/segment_anything/results/coco_vitb_expanded_18images/`.

2. **SAM expanded measured results**: all-59 mean mask IoU **0.77732861175**, mean Dice **0.85788043966**; 35-paired mean SAM IoU **0.81835742403** vs GrabCut IoU **0.62150559368**, mean paired difference **0.19685183035**. Two successful independent real-artifact bootstrap workflows: https://github.com/JLondon11/AppliedML/actions/runs/41981632551 and https://github.com/JLondon11/AppliedML/actions/runs/41981873865. Image-cluster bootstrap, 10,000 resamples, seed 20261104, paired difference 95% interval **[0.10961294,0.28484958]**; all-SAM mean 95% interval **[0.72804707,0.81959561]**. These are conditional exploratory intervals for eligible scenes, **not official COCO AP or representative full-COCO confidence intervals**.

3. **Original COCO-pretrained Faster R-CNN on all original Penn–Fudan photographs**: https://github.com/JLondon11/AppliedML/actions/runs/41980566831. All **170** original annotated images, **423** ground-truth persons, **1199** retained boxes at confidence>=0.01, **417** matched TP at IoU>=0.5, local AP50 **0.9681834843**, precision **0.3477898249**, recall **0.9858156028**. Original MobileNetV3 320-FPN weights SHA256 `907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5`. The first 12-image pilot used cutoff 0.25, so raw operating-point precision differences are confounded; no CrowdHuman/CityPersons occlusion evaluation is implied.

## Manuscript and source deliverables

The user-delivered source ZIP `Chapter_5_Expanded_SAM_and_Full_PennFudan_20261104.zip` includes revised **105-page** Chapter 5 PDF, complete LaTeX/figure assets, actual raw CSV/JSON prediction outputs and figures, existing source code, locally exercised `sam_bootstrap.py`, and a comprehensive detailed QA report.

- **Table 5.14 and Figures 5.48–5.49:** independently executed expanded SAM and image-cluster uncertainty, including original four-panel COCO mask predictions.
- **Table 5.16:** measured full 170-image Faster R-CNN comparison against original 12-image preliminary run, with different confidence cutoffs declared.
- 51 figure environments, 16 table environments; two final LaTeX passes, 105 pages, no unresolved citations/undefined references or overfull-line warnings. Rendered pages reviewed and figure asset paths verified.
- The refreshed LaTeX/PDF delivery is available in the local source archive; **this QA record does not claim the entire revised manuscript PDF was pushed to GitHub**.
- All associated experiment scripts and GitHub workflows are committed in the book's Chapter 5 case-study directories, and workflow artifacts preserve original full inference results.

## Remaining named-model Master QA blockers

Actual trained KITTI PointPillars inference and official 3D AP; agricultural-field DeepAnomaly detector; UAV video object tracking with IDF1/HOTA, trajectories and bounded closed-loop navigation; synchronized real camera–LiDAR fusion benchmark; trained TuSimple lane detection model and official lane metrics; real CrowdHuman/CityPersons occlusion-stratified trained detector benchmark; actual trained fastMRI VarNet or Adaptive-CS-Net reconstruction; and matched original multi-view NeRF/3D Gaussian Splatting GPU benchmark. Source-level classical studies, pretraining-transfer baselines and unit tests do NOT substitute for the required named experimental evidence.

**These remaining studies and full official SAM segmentation generalization remain BLOCKED.**
