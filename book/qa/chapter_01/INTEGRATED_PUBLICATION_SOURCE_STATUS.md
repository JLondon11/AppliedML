# Chapter 1 Integrated Publication Source — Production Status

Branch: `book-publication-recovery-final`

## Current integrated source

A new clean integrated Chapter 1 source has been generated from the recovered Pass20 editable baseline plus the frozen final Chapter 1 section and figure set. It is intentionally treated as a **new integration artifact**, not mislabeled as the incomplete historical integrated-v2 source.

- Source SHA-256: `9cc48d6a76a8edf59a4ec1988f528ee84ff11ca9471b389f579f94c4c50e1ea6`
- Gzip SHA-256: `3dd6579e0e6ffd39b88ca27faf3c23f4d59a632793b8684c1de1f377500b6603`
- Editable source size: 268,013 bytes
- Compressed source size: 83,363 bytes
- Base64 transfer size: 111,152 characters

## Integrated content inventory

- Retained scientific figures: **22**
- Retained substantive numbered code listings: **15**
- Full Applications: **5**
  - Predictive Maintenance for Industrial Rotating Machinery
  - Medical Image Classification
  - Financial Fraud Detection Under Extreme Class Imbalance
  - Smart Grid Forecasting and Energy Systems
  - Aerospace Structural Health Monitoring and Turbofan Prognostics
- Consolidated analytic Case Study: **Cross-Domain Optimizer Selection**
- Earlier thin optimizer domain examples are retained only as **Worked Examples**, not full Applications.
- Obsolete autonomous-driving, climate/weather, edge-AI, and precision-agriculture full Applications were removed from the Chapter 1 Applications section rather than retained without final evidence packages.

## Integration corrections completed

- all 22 frozen final figure families are explicitly wired into the integrated LaTeX source;
- the medical preprocessing figure is now discussed and integrated in the medical-imaging Application;
- all retained figure references use LaTeX labels rather than hard-coded historical figure numbers;
- the malformed CNN optimizer listing was replaced with a complete runnable training/validation listing;
- seven additional substantive listings were added to bring the chapter to the hard 15-listing minimum:
  - train-only preprocessing;
  - finite-budget random hyperparameter search;
  - nested cross-validation;
  - split-conformal regression interval;
  - selective-classification risk-coverage curve;
  - expected calibration error;
  - reproducibility manifest;
- all listings use the shared `BookCode` style, line numbering, descriptive captions, and stable labels;
- reader-facing `Illustrative Engineering Example` prefixes were removed;
- stale hard-coded references to superseded historical figures were removed.

## Structural compile

The exact integrated source was compiled locally with LuaLaTeX using a Springer-like monograph trim for structural QA. Figure artwork was represented with dimension-matched placeholders **only for this structural compile**; this is not the final publication proof.

- Trim: 6.1 × 9.25 in
- Structural PDF pages: **91**
- Target range: **90–130 pages**
- Source compiles successfully.
- The final proof must be compiled from the repository with the real accepted PDF figure assets before Chapter 1 is marked publication-final.

## Remaining Chapter 1 closure steps

1. Complete transfer/materialization of the integrated source into the repository root.
2. Compile against the real 22 accepted PDF figure assets.
3. Run reference/listing/figure/page-layout QA on the compiled proof.
4. Render and inspect the final PDF for clipping, whitespace, float placement, figure/caption fidelity, and problem-set layout.
5. Commit the publication PDF and final editable TeX to the Chapter 1 root.
