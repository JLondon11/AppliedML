# Chapter 5 — original SAM/Mask R-CNN archive content verification (2026-11-12)

Governing proportional experimental policy: repository root `SCIENTIFIC_EXPERIMENT_CERTIFICATION_STANDARD.md`. **Result: meaningful R9 evidence completed, but R9 remains PARTIAL pending final manuscript/asset linkage and durable retention. No false certification.**

## Original executed artifacts downloaded and actually inspected

The GitHub Actions workflow artifact ZIPs were retrieved from their exact IDs and inspected directly in the local analysis runtime.

| Model experiment | Original Actions run | Artifact ID | Archive SHA-256 | Byte count | Files | Archived PNG outputs |
|---|---|---|---|---:|---:|---:|
| SAM ViT-B seeded selected original COCO images | 41980537230 | 13394195846 | `b603229a59fee731eb0e4c794e34b0d776c23492a64e7cf8d6b69db16e3b3f6d` | 36,239,351 | 31 | 9 |
| SAM ViT-B first eligible selected original COCO images | 41981250063 | 13393449670 | `3a4304d56d8d7ad2eebf880069f2fc35afc79f5c78679eb94a2c36a192ed5579` | 28,593,183 | 25 | 7 |
| Original torchvision Mask R-CNN same seeded COCO images | 41981859101 | 13394521613 | `85ceb81911ff0f69975d1177f928d98be0e82c67f17d3df361fbde2bb92beacb` | 40,049,948 | 27 | 8 |

Each original archive contains actual `per_object_metrics.csv`, `selection_provenance.json`, `metrics_summary.json` and original model-result figure files (PNG/PDF/SVG). The original selection provenance lists the observed JPEG file SHA-256 values. The ZIP archives do **not** establish that raw COCO images or pixelwise output masks themselves are permanently redistributed, and those copyrighted source photographs are not claimed to be in the repository.

## Independent calculation from per-object measured records

- Original SAM seeded run: **18 unique scene IDs, 59 distinct annotation IDs**; mean SAM IoU **0.7773286117506403**, mean Dice **0.8578804396587679**, 35 valid same-box GrabCut pairs.
- Original SAM ascending run: **24 unique scene IDs, 88 distinct annotation IDs**; mean SAM IoU **0.7794836658085079**, mean Dice **0.8575854772506294**, 64 valid same-box GrabCut pairs.
- **42 completely disjoint original image IDs, 147 unique original evaluated objects** across the two runs.
- Pooled mask IoU **0.7786187121390236**, mean Dice **0.8577038635232836**; **99 valid paired objects**; mean paired SAM IoU **0.8062725695842572**, GrabCut IoU **0.48647709767168457**, observed paired difference **0.3197954719125727**.
- Original Mask R-CNN: **18 source image IDs**, **the identical 59 image/annotation ID pairs** and identical source image hashes as the seeded SAM run, with **56 actual matched instances** and per-object mean IoU **0.7453008745762046**.
- The original summaries reconcile to numerical precision. This is *independent measured-record recomputation*, not new model inference or re-running SAM weights.

## Scientific figure inspection

Two actual original archived outputs were extracted and inspected visually:
1. `Figures/sam_coco_53529_14008.png`: four panels showing the original COCO photograph/dog box and differing real-image masks. All images depict a consistent scene; colored predicted/GT mask treatments are distinguishable. Figure requires an accompanying in-text explanation identifying exact panel semantics.
2. `Figures/sam_vs_grabcut_iou.png`: a clear same-ground-truth-box-prompt IoU scatter, 0–1 axes and y=x reference. It is visually informative and avoids bar charts, but each marker reflects a subset of the original paired objects rather than the entire COCO population.

Only these selected two original figure PNGs were visually inspected in this pass, **not all 24 archived PNG outputs** or the entire manuscript.

## R1–R10 decision

The preceding proportionally scoped SAM reassessment `CH5_LEVEL_B_REASSESSMENT_CS03_CS09_CS11_20261112.csv` remains **9 PASS + 1 PARTIAL** (R9). This archive-level verification closes the missing *archive existence, original per-object count reconciliation, same-image ID matching, source-hash inventory and the verification code path* parts of R9.

R9 is **not promoted to PASS** pending: detailed source/version/checkpoint correspondence audit across the original GitHub workflows; independent verification of every claimed result figure's original source table/plot logic; durable storage/preservation before original Actions archive expiry; and exact source figure-number/page/caption confirmation against the latest compiled Chapter 5 PDF and TeX. The full latest Chapter 5 release artifact is not currently present in the local working environment.

No official COCO segmentation AP, prompt-free model evaluation, representative all-class generalization, or automobile autonomy claim is inferred. Mask R-CNN and SAM have unequal input information. The figure/source evidence supports **offline prompted real-image segmentation** only.

## Reproducibility

GitHub code `qa/verify_original_sam_archives.py` accepts `--sam-18`, `--sam-24`, `--maskrcnn-18`, `--output`; it rejects original-image/annotation mismatch and metric inconsistency. A local source-equivalent archive audit script was executed successfully against the three downloaded original ZIPs, writing `verified_original_sam_archives.json` with the actual ZIP hashes. Neither the status matrix nor any other unit has been promoted by executing this verifier.

**Decision: Chapter 5 remains scientific HOLD. The real SAM experiment's per-object numerical evidence is now directly verified at archive level, with manuscript-integrated R9 remaining open.**
