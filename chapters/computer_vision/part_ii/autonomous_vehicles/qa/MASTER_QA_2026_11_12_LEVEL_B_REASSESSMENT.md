# Chapter 5 — proportional Level B evidence reassessment (2026-11-12)

Governing standard: repository root `SCIENTIFIC_EXPERIMENT_CERTIFICATION_STANDARD.md`.
This is a **fresh assessment**, not a replacement for the preserved historical strict matrix.
Unit-level judgments are recorded in `CH5_LEVEL_B_REASSESSMENT_CS03_CS09_CS11_20261112.csv`.

**Outcome:** CS09 SAM is 9/10 PASS and R9 PARTIAL **within the precisely stated offline oracle-box prompt regime**; the other two units remain clearly blocked. **No new unit receives a final scientific certificate** from this audit. All other Chapter 5 units retain their preexisting HOLD status pending fresh review and source experiments.

## CS09 — SAM zero-shot instance segmentation — Level B

**Scope:** zero-shot means the original SAM ViT-B weights were frozen; evaluation was on original COCO images with **ground-truth box prompts supplied**, NOT unprompted automatic segmentation. This is an offline, selected eligible COCO-image evaluation, NOT an official COCO benchmark, a demonstration of full small/crowded-object generalization, or an autonomous-driving study.

- **R1 PASS:** original COCO 2017 validation JPEGs and original instance masks, original archive and model SHA256; run references in `qa/MASTER_QA_2026_11_05_SAM_42_AND_MASKRCNN.md`.
- **R2 PASS for the actual frozen-inference methodology:** no SAM training/fine-tuning happened in this experiment, so inventing training and validation sizes would be wrong. Two original selected evaluation subsets had 18 and 24 **nonoverlapping images**, with 59 and 88 annotated objects, respectively. SAM's historical training-set overlap with this sampled COCO evaluation is **not independently excluded** and must be explicitly disclosed; this PASS is for a controlled *pretrained-model illustrative evaluation*, NOT evidence of independent zero-shot generalization to an unseen source domain. If the manuscript claims independent held-out generalization, revert R2 to PARTIAL.
- **R3 PASS:** actual official SAM ViT-B inference; three successful original-run workflow references and checkpoint hash.
- **R4 PASS:** actual generated masks, per-object measured metrics, real-image overlays and retained model-result artifacts were reported by the executed workflow; the full archived objects must still be verified for R9.
- **R5 PASS:** 99 valid original boxes have meaningful matched SAM-versus-GrabCut mask IoU under common box-prompt information. The same-scene Mask R-CNN results are an *exploratory unequal-information comparison*, not a fair architectural ranking. CS09 is **not** claiming superior performance over unprompted automatic Mask R-CNN.
- **R6 PASS:** measured IoU/Dice, scene-cluster bootstrap 10,000 draws, explicit 95% exploratory intervals, the missing 48 GrabCut pairings acknowledged and test-sample selection limits recorded. Do not treat the selected-scene interval as a representative COCO-population interval.
- **R7 PASS for offline prompted segmentation only:** genuine original photographs and ground-truth instance masks directly assess the task. Real-vehicle timing, physical autonomy, clinical validation and deployment are **not** claimed in this scoped unit.
- **R8 PASS at figure-artifact level:** genuine original COCO mask overlays and measured non-bar dot-interval scientific figure with sample sizes and uncertainty. The current published manuscript page/asset linkage remains unverified under R9.
- **R9 PARTIAL:** executable sources/working model configuration, checkpoint hashes and reported original workflow artifacts exist, but a complete independent audit of the **full original per-object outputs, dataset artifact image hashes, precise current manuscript figure-to-code linkage, and durable retrieval** has not yet been completed. An archived asset mentioned in a README is not automatically a fully audited release artifact. Final source-to-figure certification remains blocked.
- **R10 PASS only with the above narrow wording:** conclusions are limited to **original, non-finetuned SAM segmentation with oracle box prompts on selected real COCO images**; no official AP, general all-class superiority, prompt-free result or autonomous perception claim.

**Measured results:** 42 scenes, 147 annotated masks, original SAM IoU mean 0.7786187121 and Dice mean 0.8577038635. The 99 valid paired masks show SAM IoU 0.8062725696, GrabCut IoU 0.4864770977 and paired difference 0.3197954719 (exploratory scene-cluster interval [0.2135099225,0.4130896346]). These are *previously executed genuine results*, not measurements rerun in this pass.

**Final promotion conditions:** inspect original individual-mask artifacts and original scene ID lists/checksums; verify the figure pixels and numerical source tables; inspect the latest actual Chapter 5 PDF/TeX figure mapping and exact statement of the title/claims; independent reviewer signs off R9. Until then **HOLD**, despite nine favorable requirement assessments.

## CS03 — YOLO autonomous drone — Level B

Original COCO-pretrained YOLO11n single-frame model transfer to 24 original VisDrone validation photographs is real. It yielded 972 original annotated objects, 2,589 retained detections, 322 matched objects, local macro AP50 .1406456330 and cutoff recall .3312757202; the measured non-bar figure was separately visually audited.

However, the current **autonomous drone** title entails more than detection. There is no original chronological tracking performance (identity ground truth, IDF1/HOTA, switches), control/planner action traces, hardware-grounded operating latency or controlled safety evaluation. These are substantive R3/R4/R6/R7/R10 gaps. The reported benchmark also lacks a matched model comparator (R5), and image selection/overlap and full original result archival inspection remain incomplete. **Do not certify** as the full title. A separate, honestly retitled Level B *single-frame aerial object detection transfer* application could be reconsidered, but the current title must remain blocked without tracking/control.

## CS11 — pedestrian detection under occlusion — Level B

All 170 authentic Penn–Fudan images and 423 annotated persons were evaluated with a pretrained Faster R-CNN and original photo overlays; local AP50=0.9681834843. This is real evidence, not synthetic output. However, Penn–Fudan is not a sufficiently annotated occlusion-specific benchmark, and the named application is **under occlusion**. A CrowdHuman/CityPersons original annotation-specific model run, stratified missed-person/FP analysis, credible occlusion baseline and measured figure are necessary. General detection results cannot certify the occlusion title. R3/R4 are PARTIAL because a relevant model executed, but not on the titled task's original occlusion dataset. **HOLD.**

## Book-wide policy safeguards exercised

1. All ten R1–R10 requirements were evaluated; no unjustified N/A was granted.
2. No requirement was waived for missing inputs or convenience.
3. Same-prompt SAM/GrabCut is a genuine paired baseline; unequal-information SAM/Mask R-CNN is explicitly exploratory.
4. Statistical interpretation is kept proportional: conditional original-image-cluster resampling is sufficient for this limited COCO demonstration; no unsupported general-population confidence claim.
5. No new results, datasets, images, checkpoints or final manuscript outputs were invented.
6. Historical strict register `CH5_R1_R10_EVIDENCE_MATRIX_20261112.csv` remains unchanged.

**Next evidence priority:** retrieve/independently audit complete original SAM workflow artifacts, current manuscript PDF and actual figure placement, then resolve R9 and issue or withhold its Level B certificate. For CS03 and CS11 the missing actual task-specific experiments cannot be removed through document edits.
