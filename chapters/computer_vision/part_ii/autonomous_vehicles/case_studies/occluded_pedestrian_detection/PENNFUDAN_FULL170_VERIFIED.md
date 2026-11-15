# PennFudanPed full original image-set experiment — verified subexperiment

- Completed model run: https://github.com/JLondon11/AppliedML/actions/runs/42458502527
- Dataset: original PennFudanPed, all **170** original photographs and original instance masks (not sampled).
- Model: torchvision Faster R-CNN MobileNetV3-Large 320-FPN, official COCO-pretrained checkpoint SHA256 `907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5`.
- Model is **not** trained on PennFudan; inference only. Detector confidence >=0.01; IoU >=0.50 one-to-one matching.
- Reference pedestrians: **423**; predicted person boxes: **1,199**; matched reference people: **417**; missed reference people: **6**; unmatched predictions: **782**.
- **Local AP50: 0.9681834843380837**; precision: **0.347789824854045**; recall: **0.9858156028368794**.
- Full original source SHA256 manifest: **170 distinct image names**, recorded in run artifact. Original output CSV has **1,199 predictions**. Precision and recall independently recalculated from counts.
- Run artifact `chapter5-pennfudan-full170-FRCNN-inference` contains `metrics.csv`, `detections.csv`, `provenance.json` and six original-image diagnostic detection panel figures.
- Section-specific chart generator: `plot_pennfudan_full170.py` checks the actual artifact's counts and generates scientific PNG/PDF/SVG.
- Manuscript integrated as Table 5.24, Figure 5.58 (Pass 16, 119-page PDF).

This verifies the complete original PennFudan detection subexperiment, **not** a full CrowdHuman or CityPersons pedestrian-occlusion unit. PennFudan masks do not independently annotate full-body versus visible-body extents; the task-specific occlusion benchmark remains blocked. The earlier 12-image diagnostic used a different confidence threshold and cannot be numerically compared at identical operating points.
