# Chapter 5 — original SAM evidence preservation and measured failure analysis

**Date:** 2026-11-12. **Scientific decision:** original per-object numerical evidence preservation PASS; new real-data failure analysis produced; **final R9 and Chapter 5 remain HOLD** pending authentic latest manuscript figure/claim audit and any remaining code/asset checks.

## Newly resolved evidence-retention obstacle

The original three GitHub Actions inference archives (18-scene SAM, 24-scene SAM, matching 18-scene Mask R-CNN) were previously unexpired but were scheduled for deletion on 2027-02-03. A new GitHub Actions preservation workflow **actually populated the repository**, and the connected GitHub API has **independently retrieved every one of the following 12 committed files**:

- `case_studies/segment_anything/results/original_actions_preserved/sam_18/`: original `per_object_metrics.csv`, `metrics_summary.json`, `per_class_summary.csv`, `selection_provenance.json`.
- `case_studies/segment_anything/results/original_actions_preserved/sam_24/`: the same four original experimental records.
- `case_studies/segment_anything/results/original_actions_preserved/maskrcnn_18/`: original `per_object_metrics.csv`, `metrics_summary.json`, `selection_provenance.json`.
- `case_studies/segment_anything/results/original_actions_preserved/SHA256_ORIGINAL_RECORDS.json`: pinned expected SHA-256 for every one of the 11 byte-identical original records.

The workflow downloads original GitHub runs 41980537230, 41981250063 and 41981859101, checks each original record's pinned SHA256 and original scene/annotation identities and numeric totals, and commits only validated numerical source/provenance files. **It does not upload copyrighted COCO photographs, raw mask PNGs or licensed model weights** into the repository. Existence of every expected file was verified with independent GitHub fetch_file reads; the hash catalog was fetched and checked against the original previously examined artifacts.

**Original results preserved** (not re-executed at this step): 42 disjoint original COCO scenes, 147 prompted SAM masks, 99 valid same-box SAM–GrabCut pairs; matched Mask R-CNN on the exact 18-scene set and 59 annotation IDs. Mean SAM IoU=0.7786187121390236, Dice=0.8577038635232835; paired mean delta=+0.3197954719125727.

## New real observation-level failure analysis

Unlike a chart of only summary means, original `per_object_metrics.csv` for the two original SAM runs was read directly and an additional premium-palette figure was executed locally:
`sam_original_object_level_error.pdf/.png/.svg`.

Scientific panels:
- **A** plots all 147 observed SAM IoUs against each object's genuine original COCO annotated area (logarithmic horizontal axis), distinguishing the independently selected 18-scene and 24-scene runs.
- **B** plots the 99 individual SAM-minus-GrabCut IoU differences against genuine original object area, with zero/equality line and measured paired-mean reference.

The previously verified pooled mean SAM IoU is 0.7786187121, but **17 of 147** original recorded object masks had IoU **below 0.50**. In the paired same-box subset, SAM obtained higher IoU on **82/99** objects and was **not better on 17/99**. The observed paired difference ranged from approximately **-0.3336 to +0.9115**. These discrepancies underscore the need for genuine failure analysis rather than a generic assertion of architectural superiority. The selected-area sample and oracle GT box prompt limit generalization. No original-model inference was rerun; all points are observed model results against original ground truth.

The publication-ready figure is a source-data scatter/paired-error graphic rather than a bar chart. Local PNG was visually inspected and the single-page PDF was rendered and inspected. Original user-download evidence bundle includes the 11 copied, SHA-pinned, original measured records, exact plotting script, figure assets and extracted statistics. It omits copyrighted original photographs and weights.

## GitHub implementation

- `.github/workflows/chapter5_sam_preserve_original_numeric_evidence.yml`: download, validate exact original numerical files, and commit in reproducible source layout.
- `figure_generation/real_sam_uncertainty/plot_original_instance_failure.py`: regenerate object-level observed scientific figure and measured failure statistics from committed original per-object CSV.
- `.github/workflows/chapter5_sam_real_object_error.yml`: automated source-traceable regeneration of the real observation-level figure.
- `case_studies/segment_anything/README.md`: expanded to correctly foreground current 42-scene experiment and limitations rather than leaving the original 3-scene pilot as apparent main result.

## Certification boundaries

This pass resolves the earlier concern that **source numerical records were accessible only through expiring GitHub Actions ZIP archives**. However, current **R9 remains PARTIAL**: although numerical measurements, source hashes and code lineage are demonstrably preserved, the latest 108-page or later Chapter 5 manuscript TeX/PDF still has not been obtained and read. The original full output artwork is still referenced by external archive artifacts; not all model output figure files were moved into a durable repository asset release. The current printed figure numbers, captions, discussion and legibility cannot be certified.

**CS09 SAM remains 9/10 proportional Level B PASS with R9 PARTIAL and full Chapter 5 scientific QA HOLD.** Other units and missing model-specific experiments are unchanged. No new invented data were introduced.
