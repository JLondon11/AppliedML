# CV5-EXP-CROWDHUMAN-96-OCCLUSION-003 — CERTIFIED (bounded experiment)

**Date:** 2026-11-08. **Certificate level:** a specific real-data computational experiment, **not** a full case-study/application approval or official CrowdHuman leaderboard result.

**Bounded claim:** A genuine COCO-pretrained torchvision Faster R-CNN ResNet50-FPN person detector was executed on 96 original CrowdHuman validation photographs containing 2,296 eligible full-body reference persons. One-to-one IoU50 person-detection recall was measured at the study confidence cutoff of 0.30, stratified into low, medium and heavy **bounding-box area proxy** occlusion groups using paired visible and full-body annotations.

**Original model execution:** https://github.com/JLondon11/AppliedML/actions/runs/42323774125  
**Archived evidence SHA256:** `02b77f17c89454989949b3d807f2310941f6c6f9397a7a526a5a2c5da444f1d3`  
**Checkpoint SHA256 recorded by the inference run:** `18b9a56960b73094552b11209ca418925cbc169236196b32f274a8fc77e82590`.  
**Source:** original CrowdHuman ODGT annotations and corresponding validation images, not synthetic detections.

| Local recall measurement | Numerator | Denominator | Recall | Image-clustered 95% CI |
|---|---:|---:|---:|---|
| Overall eligible people | 1070 | 2296 | 0.46602787 | [0.39208, 0.56068] |
| Low occlusion proxy (<0.2) | 653 | 861 | 0.75842044 | [0.63444, 0.86694] |
| Medium (0.2 to <0.5) | 325 | 638 | 0.50940439 | [0.41457, 0.62948] |
| Heavy (>=0.5) | 92 | 797 | 0.11543287 | [0.08943, 0.15200] |
| Low minus heavy (absolute) | — | — | 0.64298757 | [0.52261, 0.75065] |

**Independent audit:** every per-person result was checked for valid bounding-box geometry, 0/1 matching label, valid proxy range and group assignment. The 2296 source rows span precisely 96 distinct images. Group numerators/denominators reproduce the archived published-data-independent summaries. The five uncertainty intervals were recomputed using **5000 image-clustered bootstrap draws**, seed 20261107, with every image's annotations resampled as a cluster. See the committed `verify_crowdhuman_certificate.py` and the original run's source script and per-person artifact. The manuscript contains the independently computed occlusion equation, table, original-prediction-derived figure, scientific interpretation and limitations.

**Ten bounded-experiment criteria:** 1 PASS (appropriate real CrowHuman pedestrian occlusion proxy task); 2 PASS (original validation subset and explicit subset caveat, no local training); 3 PASS (original pretrained detector executed); 4 PASS (96 images, 2296 persons, trained-offline checkpoint, score/IoU and proxy thresholds); 5 PASS (local detection recall and three explicit scientifically motivated strata as direct comparisons); 6 PASS (code-generated measured occlusion-response figure, accompanying table); 7 PASS (source, executed logs, output CSV and checkpoint hash); 8 PASS (original measured results separated from official CrowdHuman benchmark); 9 PASS (image-cluster uncertainty, sensitivity/size/confounding limitations); 10 PASS (independent row-by-row arithmetic, plotted percentages and manuscript text concordance).

**Scientific limits:** Box-area proxy is not visible person-pixel fraction; the pronounced correlation between proxy occlusion and recall is *not* a causal estimate independent of density/size; 96 images are not complete CrowdHuman validation. This experiment provides no precision/AP, log-average miss rate, Jaccard index, CityPersons 'reasonable' benchmark or multi-model occlusion baseline. These must not be inferred. Official or general deployment claims, and the broader Chapter 5 occluded-pedestrian unit, remain **PENDING**.

This is an internally audited scientific experiment certificate, not independent third-party accreditation.
