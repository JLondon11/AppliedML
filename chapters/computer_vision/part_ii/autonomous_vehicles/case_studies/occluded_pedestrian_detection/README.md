# Chapter 5 — Occluded-pedestrian detection

**Master QA: BLOCKED**. This repository includes an executable real-image model evaluation pipeline, but no pretrained checkpoint or CrowdHuman original images were available in the current runtime; there are **no measured checkpoint-inference results**.

Dataset: [CrowdHuman](https://www.crowdhuman.org/download), official `annotation_val.odgt`. The ODGT contains `gtboxes` with `fbox` full-body, `vbox` visible-body, `tag` and optional `extra.ignore`. Paper: Shao et al., CrowdHuman: A Benchmark for Detecting Human in a Crowd, arXiv:1805.00123. Model: [torchvision Faster R-CNN ResNet-50 FPN](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html), **COCO-pretrained** checkpoint, with exact framework version / weights SHA recorded.

```bash
pip install numpy opencv-python torch torchvision
python evaluate_crowdhuman_frcnn.py \
 --image-dir /data/CrowdHuman/Images \
 --annotations-odgt /data/CrowdHuman/annotation_val.odgt \
 --checkpoint /models/fasterrcnn_resnet50_fpn_coco.pth \
 --output results/
```

Results include actual model overlays (full-body GT vs predicted boxes), per-person matching, three occlusion-stratified recall rows, and reproducibility manifest. Occlusion is a **proxy** 1-visible bounding-box area/full-body bounding-box area, not pixel visibility or CrowdHuman's native occlusion labeling. Local recall (IoU50 full-body GT) must **not** be called official CrowdHuman mAP, MR or JI. No checkpoint or dataset distributed in the repository. No model results fabricated.

Notes: the checkpoint must be an exact 91-class torchvision COCO Faster R-CNN state dictionary, loaded with strict=True. Unit tests exercise algorithms only, not neural inference. CrowdHuman is real crowded pedestrian imagery; evaluation on it alone does not certify autonomous-highway safety.

## Evidence preservation and results figure

The evaluator now writes `all_person_predictions.csv` (all retained detection boxes and scores), `original_image_sha256.csv` (original per-photograph file hashes), `per_person_results.csv`, `recall_by_occlusion.csv`, six original-image overlays, and `run_manifest.json`. These support later independent metric recomputation; retain the full files with their original experiment run, without editing measured predictions.

Once real CrowdHuman inference has successfully generated its CSV, create the publication figure **only from those measured results**:

```bash
python plot_crowdhuman_occlusion.py \\
  --input-csv results/per_person_results.csv \\
  --output results/figures/ --resamples 5000 --seed 20261111
```

The script writes measured occlusion-stratified recall with **original-image-cluster bootstrap** exploratory intervals, PNG/PDF figures and a JSON manifest with source-file SHA-256. The intervals describe variation conditional on the evaluated photographs; they are not official CrowdHuman benchmark uncertainties. No figure has been generated from CrowdHuman model predictions until the real original dataset and trained checkpoint are available and evaluated. Never use artificial test fixtures as source data for a manuscript result figure.
