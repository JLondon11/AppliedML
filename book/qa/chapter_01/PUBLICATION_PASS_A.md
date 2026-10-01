# Chapter 1 Publication Pass A — Manuscript Cleanup

Authoritative manuscript baseline: Pass20 PDF, pages 1-113.
Editable recovery source was reconciled against Pass20 content and later Book Editing corrections.

## Completed corrections
- Retained code listings: 19 (within the 15-22 hard range).
- Every retained listing has a chapter-scoped label and descriptive title.
- Listing 1.3 was materially repaired: the reconstruction had swallowed source code into the caption and left an undefined `evaluate_top1_accuracy` helper. It is replaced with a complete PyTorch routine containing an explicit `top1_accuracy` implementation and a valid optimizer-memory/validation-accuracy training loop.
- Removed the page-33 L-BFGS "Tip:" text.
- Removed reader-facing "Illustrative Engineering Example" prefixes from Chapter 1 tables. Table content remains subject to provenance review; removing the prefix does not convert unsupported numbers into accepted evidence.
- Removed/reworded production-style note language around the optimizer comparison.
- Confirmed no remaining "final figure should" phrase in the current Chapter 1 publication-pass source.

## Current local source integrity
File: `chapter_01_publication_passA.tex`
SHA-256: `0c4d686d9e921a5ede2e381decd5393a166044e631cb17ee62b1e427d1527a41`

## Remaining Chapter 1 gates
- Rebuild every Application and Case Study to the agreed structure: multiple intro/background paragraphs before bold Problem/Dataset/Model/Method/Evaluation/Results/Error Analysis/Limitations/Implications labels.
- Close every Application/Case Study figure and benchmark-comparison-table coverage gap with real/cited/reproduced evidence; consolidate a section rather than pad it when meaningful evidence cannot be produced.
- Provenance-audit every quantitative table and remove unsupported values.
- Individually audit all 32 Chapter 1 figures for scientific correctness, redundancy, panel captioning, heatmap/colorbar requirements, aesthetics, and tri-format preservation.
- Verify all 19 listing references in final surrounding prose after structural edits.
- Compile and re-measure chapter length; final target is 90-130 pages.
- Promote final editable `.tex` and all retained PNG/SVG/PDF figure binaries into `book/final/01_Foundations_of_Machine_Learning/`.

No chapter may be marked publication-ready before these remaining gates close.
