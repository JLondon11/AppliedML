# CV5-EXP-TUSIMPLE-32-PAIRED-005 — CERTIFIED (bounded original-data experiment)

**Date:** 2026-11-08. **Experiment:** Actual original TuSimple test-partition lane-marking segmentation on exactly 32 photographed annotated road scenes, comparing the genuine externally pretrained Keras/Torch-backend U-Net checkpoint against a predeclared, non-learning HSV white/yellow lane heuristic with a fixed trapezoidal road region. No local training or threshold fitting was performed.

**Executed source:** Pretrained model experiment https://github.com/JLondon11/AppliedML/actions/runs/42432540622 ; same-image original-data classical baseline https://github.com/JLondon11/AppliedML/actions/runs/42526905776 . Both runs finished successfully and archived per-frame metrics plus genuine scientific plots. **Input U-Net artifact ZIP SHA256:** `bf137c07abf58fe6e8419b7cbadf400419e62ebf1eedbe977d0732f5045965fd`. **Paired comparison ZIP SHA256:** `e022561179ded1160df4902ac4e31802c238b826a8fefa5d9a824fe2347fd2ff`. **Paired per-frame CSV SHA256:** `5abec57b65bc664cf137f908859605c8190354d8b6e3bd367362269d22f17f8c`. **Checkpoint SHA256:** `fd8360c607387fe6b9a31355da2a187eec4b896affb4695203f349a3354fc157`.

**Original data:** Mirror `dhbloo/TuSimple`, original `test_label.json` source comprising 2,782 test annotations, with only 233 corresponding photograph paths available from this mirror. Exactly 32 photo/annotation pairs were selected with recorded seed 20261107 for the original frozen pretrained experiment. For each matched evaluation, the original photograph SHA256 and GT rasterization foreground count were independently reverified. The source model was evaluated on distinct earlier 12 training-partition images only as an introductory diagnostic and their metrics are not pooled.

| Identical real-data metric, 32 test-partition images | Fixed HSV/color heuristic | External pretrained U-Net |
|---|---:|---:|
| Micro IoU versus native 4-pixel-wide reference lanes | 0.0581986225 | 0.1798302247 |
| Micro Dice | 0.1099956498 | 0.3048408508 |
| Micro precision | 0.1352588358 | 0.2061246167 |
| Micro recall | 0.0926843766 | 0.5850121007 |
| Mean per-image IoU | 0.0549220574 | 0.1765329547 |

**Paired difference:** U-Net minus HSV micro IoU = `0.1216316022`; 95% image-level paired bootstrap interval `[0.1074339809, 0.1353532807]`, 5,000 replicates, seed 20261108. U-Net produced higher per-image IoU on all 32 of the selected photographs, but the broad model ranking and generalization claims are not certified.

**Ten original-data experiment criteria:**
1. PASS: Actual task of image-space highway lane-marker segmentation on fitting original photographed highway scenes.
2. PASS **for bounded observed-photo comparison**: original TuSimple test-annotation partition, verified unchanged image/annotation pairs; no locally trained/tuned model. Historical source U-Net training membership is unknown and independent generalization is explicitly **not** claimed.
3. PASS: Named authentic externally pretrained U-Net inference and actual original-photo deterministic HSV baseline execution.
4. PASS: Exact 32 sample size, 233 accessible source-pair limitation, checkpoint hash, image seeds, fixed 4-pixel GT line rasterization, 0.50 U-Net threshold, predeclared HSV thresholds/ROI and zero local training.
5. PASS: Same-image and same-GT micro IoU, Dice, precision, recall plus meaningful low-complexity classical reference (not a similarly trained CNN comparator).
6. PASS: Original model's actual per-image output-mask overlap and paired measured plots; new chapter Table 5.22 and Figure 5.55 integrated with scientific interpretation.
7. PASS: Both source scripts and workflow committed in chapter TuSimple experiments; original and paired GitHub runs, per-image CSV, original image hashes and external checkpoint provenance recorded; independent checker committed.
8. PASS: Local raster-mask measurements separated from published model architecture leaderboard accuracies and official TuSimple lane-point metric.
9. PASS: 5,000 original-photo bootstrap replicates and failure mechanisms of both methods, shadow/marking/threshold effects and sampling limitations explicitly discussed.
10. PASS: All 32 independent photograph checksum and annotation-size matches, recomputed confusion-pixel arithmetic, micro/macro scores and confidence interval; rebuilt scientific 123-page PDF inspected on page 100.

**Critical exclusions:** The externally obtained U-Net checkpoint's past training image membership was not independently documented; test-*partition* is not synonymous with independent holdout relative to this checkpoint. Official TuSimple lane-coordinate accuracy/false-positive/false-negative benchmark, model ranking on full TuSimple test population, domain-generalized performance, a trained matched neural-network baseline and vehicle-control safety are NOT certified. The baseline is a fixed, deliberately simple HSV comparator; any claims about outperforming state-of-the-art models require a separate matched protocol.

**Disposition:** CERTIFIED only for the exact local, original-photograph paired mask-overlap experiment. This internal scientific evidence certificate is not external accreditation.
