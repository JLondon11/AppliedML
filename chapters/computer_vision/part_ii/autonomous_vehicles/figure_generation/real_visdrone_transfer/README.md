# Chapter 5 — real VisDrone measured transfer figure

This graphic uses the already executed YOLO11n real-image evaluation. It is **not** a new neural-network inference run and does **not** demonstrate tracking, detection of agricultural-field obstacles, planning, closed-loop flight, or official benchmark AP.

## Original result sources

- `../../case_studies/agricultural_uav_obstacle_detection/results/visdrone_coco_transfer_24/per_class_metrics.csv`
- `../../case_studies/agricultural_uav_obstacle_detection/results/visdrone_coco_transfer_24/summary_metrics.csv`
- Original model execution provenance: `../../case_studies/agricultural_uav_obstacle_detection/results/visdrone_coco_transfer_24/PROVENANCE.md`.

The reported experiment uses 24 genuine VisDrone validation frames, 972 annotated objects, 2,589 recorded predictions. Recomputed local macro AP50 over five annotated class categories is **0.1406456330**, and micro recall at the recorded score cutoff is **0.3312757202**. The sixth mapped category, **bus**, contains zero ground-truth instances: AP is undefined and excluded from macro. Truck has only one original ground-truth object; its AP50=0 is not a statistically stable general result.

## Regeneration

From the repository root:

```bash
python -m pip install numpy matplotlib
python chapters/computer_vision/part_ii/autonomous_vehicles/figure_generation/real_visdrone_transfer/plot_real_visdrone_transfer.py \
 --per-class chapters/computer_vision/part_ii/autonomous_vehicles/case_studies/agricultural_uav_obstacle_detection/results/visdrone_coco_transfer_24/per_class_metrics.csv \
 --summary chapters/computer_vision/part_ii/autonomous_vehicles/case_studies/agricultural_uav_obstacle_detection/results/visdrone_coco_transfer_24/summary_metrics.csv \
 --output /tmp/ch5_visdrone_replot
```

The script checks class/sample totals, class-level AP bounds, TP/FP consistency for classes with original ground truth, recomputed macro AP and micro recall. It outputs SVG/PNG/PDF scientific dot-plot figures and JSON source hashes. Input files must retain their original recorded bytes.

## Proposed in-chapter caption

**Figure 5.x.** Measured single-frame transfer of COCO-pretrained YOLO11n to 24 original VisDrone validation photographs. The plot reports local AP at IoU 0.50 by annotated class, together with original ground-truth instance counts and retained true positives/detections. The dashed line shows the macro AP50 computed over the five classes with at least one ground-truth instance (0.1406); the bus category had no annotated instances and is excluded. The recorded local recall is 0.3313 over 972 ground-truth objects. These are local cross-domain detection measurements, not official VisDrone benchmark AP or evidence of drone tracking and autonomous control.

**Certification:** valid reanalysis of committed measured numerical records, but **no Chapter 5 unit is promoted to fully certified** by this figure alone. The actual Chapter 5 LaTeX/PDF release still needs insertion, captions and independent page-by-page visual checking.
