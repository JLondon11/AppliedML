# Chapter 5 — proportional R1–R10 science certification corrected and separated from publication production

Date: 2026-11-12. Governing document: repository root `SCIENTIFIC_EXPERIMENT_CERTIFICATION_STANDARD.md` (amended 2026-11-12).

## Decision: first completed scoped scientific certificate

**CS09: SCIENTIFIC_CERTIFIED at Level B** for **offline original SAM ViT-B, no-finetune, oracle-ground-truth bounding-box-prompted instance segmentation on 42 selected genuine original COCO validation photographs**, with all ten R1–R10 deemed satisfied within that stated experimental scope. **PRODUCTION_HOLD** remains separately in force until the latest chapter PDF/TeX is actually inspected.

The earlier SAM `R9 PARTIAL` status was based primarily on missing current printed manuscript placement/durable full-asset release; these are **not universal scientific R9 blockers** under the adopted proportional standard. The original experiment's named implementation, checkpoint SHA-256, true source-dataset IDs and hashes, original action run IDs, per-object source records and 11 SHA-pinned numeric/provenance files, measured figure generator and meaningful original model-result panels are now available in the repository or explicitly linked Actions archives. This is adequate for scientific evidence traceability. Final printed figure number and PDF page are a separate *production* check. This is a determination based on previously executed, verified model outputs, **not a new model run**.

**Boundaries remain scientific and non-negotiable:**
- Meta SAM ViT-B was given **original ground-truth COCO boxes**; `zero-shot` refers to frozen/pretrained inference without task-specific fine-tuning, *not* zero input prompts.
- The nonrandom selected original sample comprises **42 distinct photographs/147 original annotation masks** with minimum eligible annotation area and at least two selected instances per chosen scene; it does not represent full COCO performance or full small-object generalization.
- Paired SAM vs classical GrabCut uses identical original box prompts for **99 eligible instances**, with observed mean difference **+0.3197954719 mask IoU** and conditional scene-bootstrap interval `[0.2135099225, 0.4130896346]`. Original SAM overall mean IoU **0.7786187121**. SAM is **not** superior on every image/object (at least 17 of the 99 eligible object pairings were equal/worse).
- Automatic Mask R-CNN has unequal input information and does **not** provide a fair architecture-ranking comparison under the original SAM GT-prompt conditions.
- Original model output overlays and aggregate/instance-level scientific plots exist; neither a new physical robotics experiment nor a full COCO AP challenge submission is required for this honestly limited Level B offline experiment.
- **This certificate does not certify a broader autonomous perception application, automatic prompt-free SAM, deployment or the whole Chapter 5 manuscript**.

## What is certified, and what is not

| Entity | Scientific certification | Manuscript production | Exact rationale |
|---|---|---|---|
| CS09 SAM, *box-prompted zero-shot instance segmentation on selected real COCO photos* | **CERTIFIED B (10/10)** | **HOLD** | Executed original model; valid matched reference; conditional errors; original figures; hashed code/observations. Final PDF layout not independently reviewed |
| CS03 titled *YOLO autonomous drone* | **HOLD** | **HOLD** | Real YOLO11n **single-image VisDrone detection** is verified as an embedded subexperiment, but parent title implies tracking/planning/control without observed traces |
| CS11 titled *pedestrian detection under occlusion* | **HOLD** | **HOLD** | Real all-170-image Penn–Fudan **general pedestrian detection** is verified as an embedded subexperiment, but lacks occlusion-stratified annotated dataset evaluation |
| CS01, CS02, CS04–CS08, CS10 | **HOLD** | **HOLD** | Original named model/data/output experiments are absent for their current scope, and current final PDF is unverified |

### Narrow subexperiment treatment (no scientific overclaim)

The original YOLO11n on 24 VisDrone real photographs and original pretrained Faster R-CNN on 170 Penn–Fudan real photographs are **verified genuine embedded scientific demonstrations**. They are eligible to be separately designated and independently assessed as honest *single-frame aerial object detection* and *offline general pedestrian detection* units, respectively. The broader parent case studies **must remain HOLD until retitled/restructured or the missing tracking/occlusion experiments are performed**. No unverified level or automatic 10/10 score is issued for those embedded units in this audit.

## Governing correction to R9

R9 requires adequate, runnable, traceable source, declared input provenance, actual per-item measured outputs, verified model and environment identification and measured-figure generation. It does **not** generally mandate a currently available final chapter PDF, every page number or preserving every original GitHub workflow ZIP forever. Missing PDF can block `PRODUCTION_APPROVED`, but it cannot independently reverse a genuine scientific certificate. Copyrighted datasets/checkpoints need not be redistributed.

## New live evidence register

The original historical `CH5_PROPORTIONAL_R1_R10_FULL_REASSESSMENT_20261112.csv` is preserved as an audit snapshot rather than rewritten. The **current live decision** is `CH5_SCIENTIFIC_VS_PRODUCTION_CERTIFICATION_20261112.csv`, validated by `validate_scientific_production_register.py`: **1 scientifically certified full named unit; 10 scientific HOLD; 11 production HOLD**. This new register correctly puts CS09 R9 **PASS** and all ten scientific R checks PASS, with a distinct `PRODUCTION_HOLD` field.

This is an evidence-consistent scientific reassessment, not a blanket relaxation of missing models, missing originals, or unexecuted applications.

## Remaining editorial release gate

Obtain latest actual full Chapter 5 manuscript PDF/TeX, inspect true figure assets and captions in context and revise if labels are small/placement wrong. This is required for book production and chapter release, **not a prerequisite to already earned SAM scientific certification**.
