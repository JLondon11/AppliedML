# CV5-EXP-PENNFUDAN-170-PAIRED-FRCNN-002 — CERTIFIED (bounded experiment)

**Date:** 2026-11-08. **Experimental claim:** two original COCO-pretrained official torchvision Faster R-CNN models (MobileNetV3-Large 320-FPN; ResNet50-FPN) executed on all 170 original PennFudanPed photographs and 423 mask-derived reference people using a shared person-score >=0.01, one-to-one IoU >=0.50 local AP50/precision/recall protocol.

**Execution and artifact:** https://github.com/JLondon11/AppliedML/actions/runs/42460370911 ; SHA256 `f95114d2a09e19ff613e3458e7d65ce150d5c6e72d9a9a72f8394f6a222759e1`. Model weights loaded from official torchvision checkpoint URLs, with unchanged pretrained weights; no Penn-Fudan training or tuning.

| Actual result | MobileNetV3-FPN | ResNet50-FPN |
|---|---:|---:|
| Full original photographs | 170 | 170 |
| Annotated people | 423 | 423 |
| Matched people | 417 | 423 |
| Retained detections | 1199 | 1988 |
| Local AP50 | 0.9681834843 | 0.9764306958 |
| Precision at score 0.01 | 0.3477898249 | 0.2127766600 |
| Recall at score 0.01 | 0.9858156028 | 1.0000000000 |

**Per-image verification:** Recomputed exactly from archived 340 per-image prediction-count rows: each model evaluated 170 unique images with identical reference counts; 166 paired images have equal matched-person counts, and ResNet50-FPN matches six additional people across four images. A 3000-draw paired image bootstrap gives a 95% percentile interval of `[0.00234178358, 0.02992705736]` for the ResNet50-FPN minus MobileNetV3-FPN person-recall difference. This is conditional on the original Penn-Fudan photo set.

**Ten criteria:** 1 PASS original people/appropriate detection models; 2 PASS original complete labeled PennFudanPed, pretrained-only evaluation without local training; 3 PASS true execution of both named models; 4 PASS full 170 images, 423 objects, confidence/IoU, original weights, bootstrap seed; 5 PASS reference-box local AP50/precision/recall with meaningful paired architectural comparator; 6 PASS code-generated per-image measured failure analysis and Figure 5.59 integrated with caption and manuscript text; 7 PASS complete committed code/workflow, source image reference and Actions artifact; 8 PASS local vs published benchmark distinction; 9 PASS matched-pair bootstrap and documented threshold/failure limits; 10 PASS independent record reconciliation and rendered chapter page inspection.

**Not certified:** occlusion-specific CrowdHuman or CityPersons performance, official COCO AP, broader vehicle/autonomous-driving benchmarks, hardware deployment, or the complete Chapter 5 occluded-pedestrian application. PennFudanPed lacks paired visible-body/full-body occlusion annotations. This certificate does **not** elevate the application to approved.

**Audit:** independent summary/per-image arithmetic, original data-source review, actual completed CI logs and archived experimental outputs, and inspection of the rebuilt Chapter 5 figure. No independent third-party accreditation is claimed.
