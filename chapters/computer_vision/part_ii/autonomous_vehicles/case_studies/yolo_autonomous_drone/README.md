# Chapter 5 — YOLO Autonomous Drone Application

**Status: real-image YOLO11n inference verified; model-specific tracking / planning still BLOCKED.**

## Real-dataset executed demonstration
- Source: [evaluate_visdrone_coco_pretrained.py](evaluate_visdrone_coco_pretrained.py) (identical code to the original executed generator).
- Official source dataset: VisDrone2019-DET-val; first 24 lexicographically sorted real images and original annotations.
- Original pretrained weights: Ultralytics `yolo11n.pt`, COCO pretrained, not VisDrone fine-tuned; checkpoint SHA-256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.
- Successful model-inference runs: [original experiment](https://github.com/JLondon11/AppliedML/actions/runs/41942041254) and [scientific-figure display rerun](https://github.com/JLondon11/AppliedML/actions/runs/41942503922).
- Result metrics source of record: [per-class numeric CSV](../agricultural_uav_obstacle_detection/results/visdrone_coco_transfer_24/per_class_metrics.csv) and [summary CSV](../agricultural_uav_obstacle_detection/results/visdrone_coco_transfer_24/summary_metrics.csv). Also retrieve the original full prediction CSV and PNG/PDF/SVG from the successful Actions artifacts.
- Measured five-class local macro AP50 = 0.1406456330, across 972 original boxes and 2,589 predicted candidates (confidence >=0.01); 322 matched instances.
- Results now included in Section 5.5 of the revised Chapter 5 manuscript, as Figure 5.34 and Table 5.4.

## Interpretation and limitations

The six candidate category mappings include pedestrian, bicycle, car, truck, bus and motor; bus had no GT annotations in the 24 images. The official ignored-region protocol was *not* applied, and the images constitute a small selected subset; the score is **not official VisDrone AP**, nor evidence of a model trained specifically for aerial imagery or agricultural fields. The figure uses confidence >=0.25 solely for legibility; the numerical AP was evaluated at >=0.01.

**Outstanding application-level QA:** true UAV tracking on chronological video with documented identity ground truth (IDF1/HOTA, switches and lost tracks), model latency under the real operating point, and safety-bounded controller/flight validation. No claim of completed drone autonomy or closed-loop performance should be made.

The preceding synthetic NMS and Kalman/PID outputs remain separate mathematical demonstrations.
