# Chapter 11 — Scientific/production gate Pass 13 (2026-11-14)

Governing standards: SCIENTIFIC_EXPERIMENT_CERTIFICATION_STANDARD.md (2026-11-12) and CODE_PUBLICATION_STANDARD.md.

## Re-executed real-data evidence
The source-equivalent reproduction of chapters/scientific_ai/code/figure_11_22_real_microscopy_candidates/run.py was executed twice independently with byte-identical measured JSON and rendered PNG.
- Real skimage immunohistochemistry 512×512 image; **189 stain-positive candidate regions**.
- Independent real UCI WDBC 569 records; train/val/test **341/114/114**; validation-selected logistic C=1; held-out AUROC **0.9957010582** and accuracy **0.9824561404**.
- **No paired labels** between microscopy image and patient records; **no annotated nucleus ground truth**. No MoNuSeg benchmark or cancer-cell classification claim is certified.
- Evidence SHA256 for measured results JSON: `93f6b460e0e3f540d578193340bbc8c04d477d99797219f871875c524104ff26`; generated PNG SHA256: `f43832c8eb3950787aba0dd7623413d4d28dbf857270cfc401a4cdc82dde1f2a`. The current evidence package is not asserted to be binary-published in GitHub.

## Production version collision
The directly inspected 177-page PDF has SHA256 `49073b974df666930e2b32a339e3e4d7b5daf297f4835abc2ad224282ed98b0a`. It contains zero literal `??` in extracted text.

**Figure 11.54 in that specific PDF describes Domain-Scale Scientific AI Platform layers.** In contrast, the rod assimilation experiment at `code/figure_11_54_real_rod_digital_twin/` describes a measured heated-rod results figure also numbered 11.54 **in a different compiled edition**. These figure identities conflict. The canonical manuscript/figure release directories have not been verified in the repository, nor has the exact matching 177-page PDF source been retrieved. The latest PDF should not be overwritten with older source.

## Separate decisions
- Nuclear Segmentation and Cancer-Cell Classification parent application: `SCIENTIFIC_HOLD`; limited unpaired demonstration executed but R5–R7 and R9 incomplete for the named end-to-end task.
- Scientific Digital Twins general claims: `SCIENTIFIC_HOLD`; individual measured lab experiment may support narrower scope.
- Chapter 11: `PRODUCTION_HOLD` until exact manuscript/PDF/source/figure-asset reconciliation and verification.
- Chapter publication: `HOLD`.

Resolve by freezing the final editable source version and exact figure inventory, producing the appropriate annotated nucleus experiment and other task-matched applications, verifying committed code/outputs, then rebuilding and reviewing the final PDF.