# Chapter 5 — figure quality, evidence and visualization audit (2026-11-12)

**Decision:** LIMITED FIGURE-LEVEL ACCEPT for the measured VisDrone dot plot and measured SAM uncertainty interval plot; **full chapter figures remain NOT CERTIFIED.**

## Scope actually reviewed
1. Live GitHub `CH5_CODE_MANIFEST.csv` (11 named applications/case studies, 19 recorded figure placements).
2. `figure_format_inventory.json` and `manifest.json` for Figures 5.20–5.30.
3. The actually available measured VisDrone transfer figure PDF, rendered and inspected. Source CSV fields independently reconciled in earlier committed QA.
4. Measured original-COCO SAM pooled-result JSON from `case_studies/segment_anything/results/coco_vitb_42images/pooled_results.json`; new vector/raster dot-interval figure generated and its one-page PDF rendered and visually inspected.

**Not available in the current workspace:** the complete latest Chapter 5 PDF/LaTeX release and pixel files for every figure in its final layout. Therefore there is no defensible claim that all chapter figures have been visually inspected or represent the best possible visualization.

## Figure-by-figure determinations where source evidence exists

- **VisDrone observed single-frame YOLO11n transfer plot:** PASS numerical/pixel quality *as a real measured transfer result*. Dot/lollipop AP50 plot avoids bars; counts, undefined bus AP (zero GT), sparse truck count (one GT), and observed 24 scenes/972 objects/2589 detections are clearly labeled. Does not establish autonomous UAV tracking/flight or field-agriculture detector validation.
- **New SAM measured uncertainty figure:** PASS for observed numeric-source correspondence and one-page visual layout. **42 genuine original COCO photographs/147 original masks**; SAM mean mask IoU .7786187121 (exploratory image-cluster interval [.7463495294,.8090720731]), mean Dice .8577038635 ([.8304908712,.8822431518]). **99 valid paired masks:** SAM IoU .8062725696, GrabCut .4864770977, difference .3197954719 (interval [.2135099225,.4130896346]). Panel design uses precise dot/interval graphics and no bars. Explicit oracle ground-truth box prompt and selected-scene limits. This **replots previously recorded original model outputs**, not a new execution. Source: `case_studies/segment_anything/results/coco_vitb_42images/pooled_results.json`. Code: `figure_generation/real_sam_uncertainty/plot_sam_measured_uncertainty.py`.
- **Figures 5.20–5.29:** repository inventory reports native/editable SVG masters for Figures **22, 23, 25, 26, 27, 28** and raster-backed SVG wrappers for Figures **20, 21, 24, 29**. Exact artwork and captions **not visually inspected in this pass**. Prior approval marks in the inventory are not fresh independent scientific certification.
- **Figure 5.30:** existing figure manifest says rejected owing to embedded text/caption and infographic grammar. Keep REJECTED until its real source, revised scientific design and layout can be reviewed.

## Scientific, informative and aesthetic requirements

- Distinguish conceptual architecture figures (permissible explanatory visuals) from observed experiment results (must derive from authentic source data and actual code/model outputs).
- Choose chart type by data and inference: masks/overlays for segmentation, photo/BEV point-cloud visualizations for detection, error maps for reconstruction, real temporal traces for tracking, precision-recall/calibration curves, spatial error distributions, image-cluster paired interval plots. **Minimize bar charts; do not replace them with worse graphics merely to meet a quota.**
- For every figure establish original dataset and split, original-image/file hashes, model/checkpoint/source code, plotted actual numbers, uncertainty/sample sizes where appropriate, meaningful axis units, non-trivial caption and in-text discussion, visually inspected PNG/PDF at published size.
- Premium muted navy/teal/copper palette, legible text, balanced whitespace, no redundant captions inside figures, consistent typography and multi-panel A/B labels.
- Any current section graphic about generated-data experiments, analytical phantoms, or published-data replots must not be described as newly measured original-data inference.

## Case-study figure requirement registry

| Unit | Current qualifying scope | Required observed-results visualization |
|---|---|---|
| CS01 KITTI 3D | No verified trained PointPillars run | Real point-cloud BEV with predicted 3D boxes/GT, errors by distance/difficulty, actual AP and latency |
| CS02 Agricultural UAV | Published replot and separate aerial transfer are insufficient | Real annotated agricultural fields, matched detector overlays and errors |
| CS03 YOLO Drone | Genuine single-frame VisDrone transfer | Retain audited dot plot; add original UAV video tracked identities, trajectories and control outcomes |
| CS04 Cross-dataset driving | Existing experiment described as generated-data | Measured model domain-shift comparisons on original held-out scenes |
| CS05 Production stack | No original measured stack traces | Real synchronized latency/failure traces and error diagnostics |
| CS06 Multimodal driving | Real-image stereo baseline only | Real aligned camera–LiDAR fusion outputs and actual BEV error analysis |
| CS07 fastMRI | Analytical phantom, no real trained fastMRI reconstruction | Real patient slices with model reconstructions and absolute error maps |
| CS08 NeRF vs 3DGS | Real photo baseline, no matched neural benchmark | Original multiview comparisons, observed rendering errors, measured quality–latency Pareto |
| CS09 SAM | Actual original COCO SAM/MRCNN outputs | Use audited measured interval figure; retain genuine source-mask overlays; expand robust prompt/domain analysis |
| CS10 TuSimple lanes | No original trained-model lane figures | Real highway photo/annotations/predicted lanes and official metric evaluation |
| CS11 Occluded people | Genuine Penn–Fudan baseline, no CrowdHuman occlusion study | Real crowded-scene failure overlays and observed occlusion-stratified interval plots |

**Count caution:** 19 section-local figure placements in the 11-unit manifest and the separate Figure 20–29 inventory are **not additive**; same figures may be counted twice.

## Release gate
Only the two individually reviewed real-result plot artifacts have figure-level acceptance. No whole-chapter all-pixel aesthetic certification, manuscript insertion, or 11/11 unit scientific certification is claimed. **Master QA remains HOLD.**
