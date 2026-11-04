# Chapter 5 — continuing Master QA source and code verification, 2026-11-03

Chapter identity: **Chapter 5, Computer Vision Part II — Autonomous Systems**.

## Completed and reproducible local checks

1. Extracted the latest Chapter 5 source from `Chapter_5_Protocol_Master_QA_20261103.zip`.
2. Executed all eight available case-study unit-test suites locally: all **PASS**. These prove only data handling/metric contracts, not model inference.
3. Removed unused LaTeX TikZ/PGFPlots package imports, styles and audit comments. No `tikzpicture` or `axis` scientific figure environments existed.
4. Added an explicit `graphicx` import; recompiled LaTeX three times. **98 pages, 45 figure environments, 12 tables**, no warnings on the final pass.
5. Programmatically compared extracted PDF page text with the previous published 98-page PDF: **98/98 pages identical**, so no scientific prose was changed by the cleanup.
6. Resolved all referenced image files (45 graphics in the source outside macro parameter `Figures/#1`) and recorded SHA-256 hashes in the release `FIGURE_ASSET_SHA256_MANIFEST.csv`.
7. Added a GitHub repository evidence checker `qa/verify_case_study_evidence.py` and associated workflow `.github/workflows/chapter5_case_study_evidence.yml`, which check the 11-case-study manifest, scripts' existence/syntax, and approval flags. **No CI execution was observed during this verification**, and its existence is not proof of model experiments.

## Deficiencies still blocking Master QA

11/11 case studies remain **BLOCKED** for named-model real-data inference and data-to-figure/table traceability. The real-data stereo example, published-result replots and generated-scene analytical studies provide useful narrower evidence but do not replace named-model results.

No datasets/checkpoints were obtained or trained-model outputs generated during this source audit. No inference metrics or figures should be invented to satisfy the publication gate. Chapter PDF updates are delivered locally, **not committed as binary manuscript artifacts** to this repository.
