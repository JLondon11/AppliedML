# Chapter 5 — UAV Detection and Tracking

**Master QA: BLOCKED pending the real dataset and checkpoint.** This directory contains model-specific executable source, but no genuine VisDrone detection/tracking experiment has been completed here.

Primary references: official [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) dataset and [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/). Obtain the original VisDrone2019-DET validation images and annotations (548 images) and a checkpoint actually fine-tuned on the ten VisDrone classes, with provenance, configuration and weights hash. Generic COCO checkpoints are deliberately rejected for class-specific VisDrone scoring.

Install: `pip install ultralytics opencv-python numpy matplotlib`.

Detection:
```bash
python evaluate_visdrone_yolo.py --checkpoint /data/visdrone_yolo.pt \
  --images /data/VisDrone2019-DET-val/images \
  --annotations /data/VisDrone2019-DET-val/annotations \
  --output results/det
```
Outputs genuine `predictions.csv`, `per_class_ap50.csv`, `run_manifest.json` and original PNG/PDF/SVG output+ground-truth figure panels. The local AP50 is **not** the official VisDrone challenge AP, particularly because ignored regions and official thresholds differ. Never present this as leaderboard mAP.

Video tracking:
```bash
python track_visdrone_video.py --checkpoint /data/visdrone_yolo.pt \
 --frames /data/VisDrone2019-VID-val/sequences/SEQUENCE \
 --output results/tracking
```
Creates genuinely inferred `tracks.csv` and annotated frames. It does **not** compute IDF1 or HOTA; use the official VisDrone identity annotations and appropriate matching protocol to add valid benchmark results.

Unit tests must not be confused with neural inference. No benchmark numbers, detections, tracking metrics or figures from the named model are claimed in the current chapter; it remains under Master QA remediation.


## Real aerial photograph transfer baseline (initiated 2026-11-04)

`evaluate_visdrone_coco_pretrained.py` and `.github/workflows/chapter5_real_visdrone_yolo_inference.yml` run an additional, **separately labeled** YOLO11n COCO-pretrained baseline on original annotated VisDrone validation photographs, recording genuine predictions, local mapped-category AP50, and annotated output figures. This uses only unambiguously paired COCO/VisDrone categories (pedestrian, bicycle, car, truck, bus, motor). It is **not** a VisDrone-fine-tuned model; unmatched classes, ignored regions and nonstandard category mapping prevent official challenge AP claims. This cross-domain aerial-image example does **not** satisfy the case study’s agricultural field obstacle-detection / DeepAnomaly inference requirement. Mark as completed only after GitHub Actions success and inspection of its artifact.
