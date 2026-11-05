# Chapter 5 — Real pretrained YOLO inference on original VisDrone UAV imagery

## Actual execution
Successful GitHub Actions runs:
- https://github.com/JLondon11/AppliedML/actions/runs/41942041254
- https://github.com/JLondon11/AppliedML/actions/runs/41942503922

Both runs downloaded the original VisDrone2019-DET validation archive, loaded the official Ultralytics YOLO11n COCO-pretrained model, actually ran inference over 24 original annotated aerial images, computed local IoU50 AP measurements, and uploaded prediction CSVs, figure panels and provenance.

The successful rerun uses 0.25 confidence cutoff only **for displayed figure panels**, not the actual numeric evaluation (0.01). The numeric metrics CSV files of both independent runs were byte-for-byte identical.

## Measured results (not fabricated)
- Five-class local macro AP50 **0.1406456330251602**, corresponding to five mapped classes with a GT-positive sample.
- Mapped original annotation boxes: **972**; predicted detections: **2589**; matched GT boxes **322**.
- Car local AP50 **0.48769179862033374**; pedestrian **0.19010663920955564**; bicycle **0.0027829313543599257**; truck **0** from one ground truth; motor **0.022646795941551737**. Bus undefined (zero GT).
- Model checkpoint SHA-256: `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.
- Dataset: original VisDrone2019-DET validation, 24 sorted initial filenames. Images and annotations hashed individually in provenance JSON. No model fine-tuning.

## Manuscript integration
Section 5.5 **YOLO Autonomous Drone Application** in Chapter 5 **Computer Vision Part II: Autonomous Systems**:
- **Figure 5.34**, real source annotations and actual model prediction output.
- **Table 5.4**, executed class-specific AP and GT/detection/match counts.
- Several evidence-based paragraphs explain dataset, model, selection, mapping, evaluation and limitations.
- Final source `Chapter_5.tex` and compiled **100-page** `Chapter_5.pdf` in the delivered user package. Two final LaTeX passes, no overfull lines or unresolved citations/references.

The named experiment is **a small real-data cross-domain YOLO transfer baseline**, NOT an official VisDrone challenge metric: official ignored-region handling and unmatched classes are not implemented. It is NOT an agricultural-field anomaly detector, video object tracker or UAV flight-safety assessment. Other case-study trained-model evaluations remain pending. **Chapter 5 NOT Master QA approved**.

The original executable script and the outputs have been copied into the correctly organized Chapter 5 YOLO case-study directory: `case_studies/yolo_autonomous_drone/`. The original historical successful workflow artifact is preserved in GitHub Actions. The delivered PDF/LaTeX manuscript remains a local download; it has not been committed in binary form to this repository.
