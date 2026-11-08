# CV5-UNIT-11-PEDESTRIAN-OCCLUSION-002 — APPROVED, claim-scoped graduate textbook application

**Date:** 2026-11-08. **Unit title:** Application: Pedestrian Detection Under Occlusion, Chapter 5, §5.8. **Level:** full instructional *application unit* approval within the stated scope; not official challenge, clinical/automotive safety, or full chapter accreditation.

## Exact educational/scientific scope

This application teaches quantification of person-detection recall under increasing **annotation-box area proxy occlusion** in original crowded photographs, comparisons between two standard pretrained Faster R-CNN model configurations on the *same* original CrowdHuman scenes, and critical interpretation of uncertainty, detection failure and limited cross-dataset transfer. The manuscript explicitly states that the proxy is not per-person visible-pixel fraction, and carefully distinguishes this application from unexecuted specialized head/part-based detectors, full CityPersons or CrowdHuman challenge protocols, and deployment safety.

**Supporting formally verified experiments:**
- `CV5-EXP-CROWDHUMAN-96-OCCLUSION-003`: 96 original CrowdHuman images, 2296 people, one genuine ResNet50-FPN trained model; low/medium/heavy proxy-occlusion recall and 5000 image-cluster bootstrap, original-output figure.
- `CV5-EXP-CROWDHUMAN-48-PAIRED-004`: 48 *other seeded* original CrowdHuman photographs, 1589 people, both actual COCO-pretrained Faster R-CNN variants evaluated against the identical annotations, threshold and IoU rule; 2500 paired image-cluster bootstrap, scientific paired-output figure.
- `CV5-EXP-PENNFUDAN-170-PAIRED-FRCNN-002`: all 170 original Penn-Fudan photographs and 423 annotated people, two frozen pretrained detectors, original AP50 and paired CI; included as an expressly independent pedestrian-domain contrast only, **not** occlusion benchmark evidence.
- Earlier 12-image Penn-Fudan visual diagnostic is labeled as a separate pilot, with limits stated.

## Ten proportional graduate-book criteria — reviewed

1. **Topic/dataset/model:** PASS. CrowdHuman original full-body and visible-box data match the stated pedestrian occlusion question. Standard pretrained detectors, not misrepresented as specialized occlusion-aware architectures.
2. **Authentic real data/partition:** PASS within limited claims. 96-image and 48-image original validation subsets correctly identified as **separate** samples. No artificial annotations, and no study claims full-population or official challenge performance. No local training on evaluation data.
3. **Actual named-model execution:** PASS. Two official torchvision pretrained Faster R-CNN configurations executed in GitHub Actions, including corrected paired run `42524129054`; 96-image original detector run `42323774125`.
4. **Dataset counts/configuration:** PASS. Original CrowdHuman full dataset splits sourced, exact 96/2296 and 48/1589 evaluated sizes, scores, model weights, image selection seed, no fine-tuning, IoU50, and proxy thresholds documented. Original Penn-Fudan sizes and its different score cutoff are separately declared.
5. **Task metrics and meaningful comparisons:** PASS. Original full-body IoU50 one-to-one per-stratum recall, measured real baseline detector on the same 48 frames, full-Penn paired AP50 only in different context. No unexecuted official MR/AP claim.
6. **Scientific code-generated result figures:** PASS. Both original 96-image and true same-frame 48-image outputs, including all three occlusion strata and per-image comparison. Real photographic prediction evidence from 96-image original run is documented separately from aggregate plotting; figures integrated with text, table and captions.
7. **Full code/provenance:** PASS. Original code, source dataset and pretrained checkpoint, successful Actions logs and hashed output artifact, corrected full-precision CSV, complete verifier committed to GitHub by section.
8. **Local/published separation:** PASS. Local recall, original dataset citations and historical CityPersons/CrowdHuman challenge metric descriptions are clearly differentiated.
9. **Uncertainty/failure/limitations:** PASS. 5000-cluster bootstrap for 96-image run, 2500 paired cluster bootstrap for same-frame 48-image run, low overall/heavy recall and uncertainty intervals, size/confounding and domain limitations. No unwarranted architectural causal claims.
10. **Numerical/manuscript integrity:** PASS. Independently recomputed every one of 3,178 paired model/person rows, 1,589 same-person pair keys, 48 original-image hashes, 4 paired bootstrap CIs and earlier 2,296-person counts; corrected round-to-boundary export problem; compiled chapter PDF 121 pages with no unresolved references or overfull boxes; visually checked original Table 5.23 and Figure 5.56 and caption.

## Results that bound the claim

On the 48-image paired sample, MobileNetV3-FPN full-body person recall was 323/1589 = 0.2033, ResNet50-FPN was 688/1589 = 0.4330. At heavy proxy occlusion, recall was 30/579 = 0.0518 versus 57/579 = 0.0984. Overall absolute paired difference 0.2297 (95% image-cluster bootstrap interval [0.1540, 0.3134]); substantially lower heavy-group recall is a measured association, not causation.

On the **separate** 96-image CrowdHuman sample, 1070/2296 eligible people were matched by ResNet50-FPN, with 653/861 low, 325/638 medium and 92/797 heavy proxy-occlusion recall counts. Do not pool these samples.

## Noncertified claims, explicitly excluded

This approval makes **no claim** of official CrowdHuman AP, log-average miss rate, Jaccard Index, full 4370-image validation leaderboard, CityPersons official 'reasonable' test performance, occlusion-specialized head/part detector inference, fielded autonomous vehicle safety, causal inference about occlusion independent of scale/crowd density, or model generalization beyond evaluated original subsets. Future additions claiming these require extra executed experiments and re-audit.

**Disposition:** This complete *claim-scoped graduate instructional application unit* is APPROVED under the revised two-level policy. This does not imply Chapter 5 as a whole is certified. Scientific certificate is internally audited, not third-party accredited.
