# CV5-EXP-CROWDHUMAN-48-PAIRED-004 — CERTIFIED (bounded experiment)

**Date:** 2026-11-08. **Scientific scope:** same-photo, same-annotation comparison of two frozen, official COCO-pretrained Faster R-CNN person detectors stratified by a *bounding-box area occlusion proxy* on original CrowdHuman validation photographs. This does **not** certify official CrowdHuman AP/MR/JI or person-detection deployment safety.

**Corrected, successful execution:** https://github.com/JLondon11/AppliedML/actions/runs/42524129054

**Corrected experiment artifact SHA256:** `5f1a3854e7c892a43ba09d1b9531a0651a70d7962a7dd77c665875d0bad60d56`.  
**Per-person CSV SHA256:** `230049b6548abbbdc9a6cdf8dd6b04a1f7a81e31199410507316b9ae7983789b`.

The first successful trial (run `42523207362`) rounded the stored proxy to eight decimals, causing three borderline persons to serialize as `0.2` despite an unrounded classification in the low group. The execution was repeated with full-precision proxy serialization. The corrected artifact passed independent validation of every group assignment and unchanged measured numerators. The original, imprecisely serialized archive is **not** certified.

**Data and model:** 48 original validation photographs (seed `20261108`), with 1,589 eligible real CrowdHuman full-body annotated persons, 3,178 genuinely inferred person/architecture evaluation rows, source `Funtion/CrowdHuman` mirror of the original dataset. MobileNetV3-Large 320-FPN and ResNet50-FPN COCO-pretrained official torchvision weights were executed independently on each identical image with score cutoff 0.30 and greedy one-to-one matching at full-body IoU 0.50. No local training/fine-tuning.

| Group | Original persons | MobileNetV3 hits / recall | ResNet50 hits / recall |
|---|---:|---:|---:|
| Low box-area proxy | 598 | 219 / 0.3662 | 465 / 0.7776 |
| Medium | 412 | 74 / 0.1796 | 166 / 0.4029 |
| Heavy | 579 | 30 / 0.0518 | 57 / 0.0984 |
| All persons | 1589 | 323 / 0.2033 | 688 / 0.4330 |

ResNet50-minus-MobileNetV3 recall difference = **0.2297042** overall, 95% paired image-cluster percentile bootstrap interval **[0.1540225, 0.3133834]**, 2,500 bootstrap draws (seed `20261108`). Conditional intervals: low **[0.2978954,0.5058457]**, medium **[0.1439038,0.3018240]**, heavy **[0.0099694,0.0835584]**. The conclusion is a within-sample detector comparison, not a causal architectural effect or generalization proof.

**Criteria 1–10:** PASS for this declared experimental scope. (1) Actual detection and paired occlusion-proxy question; (2) original annotated validation data, no synthetic ground truth; (3) both named pretrained models executed; (4) seed, frozen training status, 48 photo and 1589-person sizes, checkpoints, confidence and IoU protocol; (5) task-correct local full-body recall and meaningful same-data detector comparison; (6) actual code-generated original-image stratified and per-image scientific figures integrated in section; (7) section-organized implementation/CI run and hashed experimental data artifact; (8) official vs locally measured performance explicitly distinguished; (9) paired bootstrap and heavy-occlusion failure analysis, sampling and causal limitations; (10) independently reproduced all original-person counts, bins, numerical comparisons and 2500 bootstrap intervals, verified original Chapter 5 figure and caption.

**Independent audit:** `qa/certifications/verify_crowdhuman_paired.py`, executed against corrected artifact; complete 1589-person pairing and distinct image SHA256s checked. Chapter 5 121-page PDF rendered and visually examined at its original table/figure pages.

**Exclusions:** This does not represent full CrowdHuman validation (4,370 images), official challenge MR, COCO mAP, specialized occlusion-aware architectures, pedestrian tracking, or autonomous vehicle safety. It is not a proof of the causal effect of occlusion or backbone.
