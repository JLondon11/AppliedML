# Chapter 5 — measured VisDrone data reanalysis

**Decision:** Numeric evidence RECONCILED; Chapter 5 Master QA remains HOLD.

## Genuine original model evidence

Original-record source:
`case_studies/agricultural_uav_obstacle_detection/results/visdrone_coco_transfer_24/{per_class_metrics.csv,summary_metrics.csv,PROVENANCE.md}`.

Measured experiment: COCO-pretrained YOLO11n, **24 original VisDrone validation images**, 972 original GT objects, 2,589 recorded class detections. No trained-model inference was rerun in the present pass.

Independent recomputation from the committed rows:
- Five classes with at least one GT instance: local **macro AP50 = 0.1406456330251602**.
- Sum original GT across these categories: **972**; matched true positives **322**; thus recall **322/972 = 0.33127572016460904**.
- Bus: zero original GT, undefined AP; correctly excluded.
- Truck: 1 original GT, 0 matched; extreme sparse-category uncertainty.
- Per-class AP50: pedestrian .1901066392 (339 GT), bicycle .00278293135 (28 GT), car .4876917986 (250 GT), truck 0 (1 GT), motor .02264679594 (354 GT).

The result exposes severe transfer limitations outside cars, rather than evidence of operational UAV detection reliability or control. No generalization or official benchmark AP is inferred.

## New reproducible assets

The code is committed in
`figure_generation/real_visdrone_transfer/plot_real_visdrone_transfer.py`.
A companion `README.md` includes exact source paths, invocation and scientifically restricted figure caption.

Local reanalysis generated:
`visdrone_yolo_transfer.png`, `.svg`, `.pdf`,
and `source_metrics_verified.json`.
Inputs were transcribed from the repository CSV responses and retained byte-level SHA256 in the local JSON:
- per-class CSV: `8782178b4878ea86e8e705d5b6b8319440856f5ad971a77617ee296f781adfdb`
- summary CSV: `76bcb875e09db91fdfd4b9b577a27851d7d0fcae973f9f86d63392873a795589`

The local generator produced and visually reviewed a restrained-palette, non-bar dot plot with exact GT and TP/detection counts. The source-data integrity test also rejected a deliberately inconsistent macro AP report, demonstrating the positive reconciliation is not unconditional.

## QA boundaries

A replot of already-executed real-image model results **is a scientifically valid real-data result figure**. It does **not** establish a new model-run benchmark or complete the distinct missing YOLO drone video tracking, agricultural field anomaly detection, UAV planning, or closed-loop controller experiments.

The latest revised Chapter 5 source archive and PDF are unavailable in this current local workspace, so there was no manuscript integration or independent page-level layout certification. No new application/case-study unit is promoted to CERTIFIED. Named-model TuSimple/CrowdHuman/KITTI/fusion experiments remain blocked pending their original inputs and actual execution.

**Next:** integrate the audited single-frame transfer figure into the exact matching Chapter 5 unit when the source is available, then collect original tracked video predictions and evaluate IDF1/HOTA to resolve the task named in its title.
