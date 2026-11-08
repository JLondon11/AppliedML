# CV5-UNIT-10-TUSIMPLE-LANE-003 — APPROVED (claim-scoped graduate-level application)

**Date:** 2026-11-08. **Unit title:** Application: Lane Detection for Autonomous Highway Driving, Chapter 5 §5.8. **Approval is explicitly limited** to an instructional lane-mask segmentation application using genuine original TuSimple photographs and a measured classical baseline. No claim about independent test generalization, official TuSimple lane-point accuracy, deployment safety or model superiority over SCNN/RESA/UFLD is approved.

## Defined scientific/educational objective

Demonstrate how originally photographed highway lane-center annotations are converted into narrow reference masks, how a genuinely executed pretrained U-Net outputs foreground lane-mask predictions, and how this segmentation compares against a fixed classical white/yellow HSV lane-marking heuristic. Examine the sensitivity of thin-lane overlap, sample selection and checkpoint uncertainty while distinguishing independently published lane-coordinate benchmark figures.

**Experimental evidence:** Original frozen pretrained TuSimple U-Net original test-partition run `42432540622` (32 real test photographs) and independently executed fixed HSV comparator run `42526905776` on the exact same original frames, with independently matched per-photo SHA256 and ground-truth foreground pixel count, archived `CV5-EXP-TUSIMPLE-32-PAIRED-005`. Earlier 12-photo training-partition demonstration is introductory and is not reported as a generalization benchmark.

## Proportional textbook audit of the ten scientific criteria

1. **Topic/dataset/model alignment — PASS:** original annotated highway lanes, directly implemented image-space lane-mask detector and benchmarked classical lane-marking approach.
2. **Original real data/evaluation partition — PASS within explicitly bounded scope:** 32 distinct original photographs in the TuSimple test annotation partition, no local train/test mixing, same original image hashes for both methods. The external checkpoint's training membership is unknown and independent generalization is explicitly excluded from the scope.
3. **Named model execution — PASS:** original externally pretrained U-Net genuinely executed on the photographed road scenes; classic HSV detector independently ran from original images; no inferred or fabricated predictions.
4. **Sample sizes/configurations — PASS:** documented 12 training-partition introductory frames, separate 32 test-partition evaluation frames from 233 available image/annotation pairs (2,782 test annotation records), frozen model and checksum, input resolution, 0.50 threshold, four-pixel reference mask, seed and predeclared HSV color/road ROI thresholds; no local training.
5. **Task metrics/appropriate baseline — PASS:** identical-image micro/macro original-data mask IoU, Dice, precision and recall; deterministic non-learned conventional lane-marking reference. Not represented as official TuSimple benchmark or SOTA neural comparison.
6. **Code-generated scientific figures — PASS:** original annotation photograph and code-rasterized ground truth, genuine neural network per-frame mask-overlap results, 32-photo paired per-image scatter/diagonal comparison, integrated scientific captions.
7. **Complete code/provenance — PASS:** executable model evaluation, original dataset/checkpoint references, two completed GitHub Actions, source SHA256 per real photograph, independent checker and source-linked figure generator committed in chapter section.
8. **Measured versus published — PASS:** paper-reported SCNN/RESA and official leaderboard numbers explicitly separate from own raster-mask experiments.
9. **Appropriate statistical/failure analysis — PASS:** 5,000 original-image paired bootstraps for micro IoU difference, image-scale errors, color threshold failures, width tolerance and sampling limitations discussed.
10. **Numerical/manuscript integrity — PASS:** independent recomputation of 32 complete per-frame prediction/GT counts, all confidence intervals and model/baseline values; newly integrated Table 5.22/Figure 5.55 on inspected page 100; complete LaTeX compiled twice to 123 pages with no undefined references or overfull boxes.

## Verified measurements and limitations

On the same 32 original test-partition photographs, U-Net micro mask IoU was 0.1798302, compared with HSV heuristic 0.0581986, for an absolute difference of 0.1216316 (5000-image bootstrap 95% interval [0.1074340,0.1353533]). U-Net performed better on all 32 selected photographs at this specific four-pixel raster-mask metric. The reported interval is conditional on these original photos and the chosen baseline, and is **not** a performance estimate for the complete TuSimple population or a guarantee of training-data independence. This experiment did not compute official lane-point accuracy. No autonomous vehicle control, corner-case deployment or real-time processing safety was tested.

**Decision:** APPROVED under the graduate textbook's two-level *claim-scoped application* policy. This certificate grants no endorsement of supervised checkpoint generalization or official lane challenge leaderboard rankings, and does not approve broader Chapter 5 research case studies or complete book Master QA.

**Review evidence:** `CV5-EXP-TUSIMPLE-32-PAIRED-005.md`, `qa/certifications/verify_tusimple_paired_original_test.py`, model GitHub Actions `42432540622`, paired baseline Actions `42526905776`, scientifically reconciled 32-pair source data CSV, rebuilt Pass 21 123-page manuscript and Figure 5.55. This is internal scientific certification rather than independent external accreditation.
