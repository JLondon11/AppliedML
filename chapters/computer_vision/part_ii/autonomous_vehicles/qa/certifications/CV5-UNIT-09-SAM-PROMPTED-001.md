# CV5-UNIT-09-SAM-PROMPTED-001 — APPROVED (claim-scoped graduate textbook application)

**Date:** 2026-11-08. **Unit:** Chapter 5, Application: Segment Anything Model for Prompt-Conditioned Zero-Shot Instance Segmentation (Section 5.8, original section identifier `subsec:sam_zeroshot_application`).

**Scope of this unit approval:** a graduate-level **instructional application of box-prompted, frozen SAM ViT-B** evaluated for local mask IoU and Dice on a bounded original COCO 2017 validation-image subset, with a paired rectangle-initialized GrabCut classical baseline and image-clustered uncertainty analysis. No universal SOTA claim, official COCO AP, fully automatic detection/segmentation, autonomous vehicle deployment, or full-COCO representativeness is certified.

## Why unit certification is appropriate

The section's scientific objective and problem statement are explicitly bounded to prompt-conditioned COCO instance-mask overlap without model fine-tuning and classical baseline behavior. Its two genuine experiments comprise an eight-object, three-image visual pilot and a separate 120-object, 37-image measured study; the pilot does not stand in for the extended result. The expanded study contains 78 legitimate paired objects; 42 GrabCut-invalid cases are disclosed. The revised manuscript includes the correct computed SAM and baseline results, the actual output-mask panels, the new all-78-pair scientific comparison, interpretation, uncertainty, implementation link and limitations. The associated experiment is separately certified as `CV5-EXP-SAM-COCO-ORACLEBOX-001`.

## All ten scientific gates

| Criterion | Result | Verified evidence |
|---|---|---|
| 1. Task/dataset/model alignment | PASS | Unit renamed more precisely to prompt-conditioned zero-shot mask inference; COCO instance targets and oracle boxes match stated task |
| 2. Original real data and evaluation | PASS within scope | Original COCO 2017 validation photos and original GT annotations; explicitly not a full COCO population benchmark; unknown potential overlap with SA-1B disclosed |
| 3. Actual named model execution | PASS | Pretrained official Meta SAM ViT-B executed; successful GitHub run `42300131086`; no COCO fine-tuning |
| 4. Data counts and configuration | PASS | 37 unique images/120 scored objects, 78 paired objects; 42 baseline invalid; seed `20261105`; box prompting and mask metric defined; no local train split |
| 5. Task-specific metrics/comparison | PASS | True object mask IoU, Dice, paired 78-object GrabCut reference; no unexecuted supervised baseline claim |
| 6. Model outputs and scientific figures | PASS | Real photographed mask predictions vs original GT and the replaced all-78-pair result figure; scientific interpretations present in their section |
| 7. Executable code/provenance | PASS | Existing SAM inference pipeline, data and checkpoint SHA manifests, per-object original predictions and newly committed plot generator |
| 8. Measured vs published separation | PASS | Original SAM paper cited as context; locally measured subset not mislabeled COCO AP or published accuracy |
| 9. Appropriate uncertainty and failures | PASS | 5000 image-cluster bootstrap draws, valid-pair selection, failures, distribution and confidence-interval conditionality documented |
| 10. Numerical and publication consistency | PASS | Independently recomputed 120- and 78-object means and counts; new full-pair scatter/histogram reconstructed from original records, TeX rebuilt twice with no unresolved refs, output page visually inspected |

## Exact verified measurements

* Original original-data study: SAM mean object IoU **0.7822379222**; mean Dice **0.8605632305** across 120 original COCO annotations.
* 78 paired objects: SAM IoU **0.8076488277**, GrabCut IoU **0.5622356850**, paired mean difference **0.2454131427**, median difference **0.1420331518**.
* Model wins by object: SAM **66**, GrabCut **12**. No ties, with **42** additional SAM-scored objects ineligible for the paired GrabCut evaluation.
* 95% image-cluster bootstrap (5000 draws): SAM IoU **[0.7430,0.8199]**, paired difference **[0.1603,0.3306]**. These intervals condition on the selected subset.

## Strict qualifications retained in the textbook

1. Oracle ground-truth localization prompts do not measure automatic category detection, open-set localization, full-scene AP, or embedded deployment.
2. COCO evaluation is a seeded, capped subset rather than a prespecified representative sample of all COCO validation objects. Population generalization is **not** claimed.
3. Unverified overlap between SA-1B and individual COCO validation photographs cannot be excluded from source metadata alone; no COCO-specific supervised fine-tuning was performed locally.
4. Only a **classical** GrabCut baseline was executed. This supports a prompt-conditioned instructional application but not claims of outperforming trained supervised segmentation models.
5. A broader SAM research case study involving multiple prompt types, latency, architecture ablations or other domains would need further independent experiments.

**Disposition:** The bounded *SAM application unit itself* passes the ten proportional scientific criteria for its revised, clearly stated instructional objective. This does not certify any other case-study unit, the complete Chapter 5 manuscript, or other externally unverified claims.

**Evidence:** archived original experiment `42300131086`, `CV5-EXP-SAM-COCO-ORACLEBOX-001.md`, actual per-object CSV and figure generator `case_studies/segment_anything/plot_full78_real_pair_results.py`, corrected 120-page TeX/PDF build of 2026-11-08 (Pass 19). Scientific review is internal rather than independent third-party accreditation.
