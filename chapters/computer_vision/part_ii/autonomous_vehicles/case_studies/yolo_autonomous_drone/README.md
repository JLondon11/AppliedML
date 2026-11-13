# Chapter 5 — YOLO Autonomous Drone Application

**Status: real-image YOLO11n inference verified; model-specific tracking / planning still BLOCKED.**

## Real-dataset executed demonstration
- Source: [evaluate_visdrone_coco_pretrained.py](evaluate_visdrone_coco_pretrained.py) (identical code to the original executed generator).
- Official source dataset: VisDrone2019-DET-val; first 24 lexicographically sorted real image files and original annotations. **Original source-image SHA-256 audit identified only 23 unique image contents**, because `0000022_00000_d_0000004.jpg` and `0000023_00000_d_0000008.jpg` have identical original image bytes despite differing filenames; no independence claim is made for all 24 file entries.
- Original pretrained weights: Ultralytics `yolo11n.pt`, COCO pretrained, not VisDrone fine-tuned; checkpoint SHA-256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.
- Successful model-inference runs: [original experiment](https://github.com/JLondon11/AppliedML/actions/runs/41942041254) and [scientific-figure display rerun](https://github.com/JLondon11/AppliedML/actions/runs/41942503922).
- Result metrics source of record: [per-class numeric CSV](../agricultural_uav_obstacle_detection/results/visdrone_coco_transfer_24/per_class_metrics.csv) and [summary CSV](../agricultural_uav_obstacle_detection/results/visdrone_coco_transfer_24/summary_metrics.csv). Also retrieve the original full prediction CSV and PNG/PDF/SVG from the successful Actions artifacts.
- Measured five-class local macro AP50 = 0.1406456330, across 972 original boxes and 2,589 predicted candidates (confidence >=0.01); 322 matched instances.
- Results now included in Section 5.5 of the revised Chapter 5 manuscript, as Figure 5.34 and Table 5.4.

## Interpretation and limitations

The six candidate category mappings include pedestrian, bicycle, car, truck, bus and motor; bus had no GT annotations in the 24 images. The official ignored-region protocol was *not* applied, and the images constitute a small selected subset; the score is **not official VisDrone AP**, nor evidence of a model trained specifically for aerial imagery or agricultural fields. The figure uses confidence >=0.25 solely for legibility; the numerical AP was evaluated at >=0.01.

**Outstanding application-level QA:** true UAV tracking on chronological video with documented identity ground truth (IDF1/HOTA, switches and lost tracks), model latency under the real operating point, and safety-bounded controller/flight validation. No claim of completed drone autonomy or closed-loop performance should be made.

The preceding synthetic NMS and Kalman/PID outputs remain separate mathematical demonstrations.

## Separate narrow scientific certification (2026-11-12)

The actual frozen COCO-pretrained YOLO11n on selected original VisDrone still photographs is independently registered as **CV5-D01, Level A SCIENTIFIC_CERTIFIED** under `qa/CH5_LEVEL_A_EMBEDDED_DEMONSTRATION_CERTIFICATIONS_20261112.csv`. The experiment is restricted to *offline aerial still-image object detection*, including its local five-class mapped AP50, observed detections and original figure panels. **This does not certify the CS03 autonomous-drone application** (no executed video identity tracking or control/flight outputs). Final chapter PDF production is separately HOLD. The first 24 filenames comprise 23 unique image hashes; this duplication affects sampling/generalization interpretation but does not fabricate detections or negate the observed original-file execution.
