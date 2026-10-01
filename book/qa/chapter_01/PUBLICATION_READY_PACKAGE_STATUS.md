# Chapter 1 Publication-Ready Package Status

Chapter: Foundations of Machine Learning  
Authoritative manuscript baseline: Pass20 PDF, pages 1-113.

## Completed and repository-persisted
- Updated Master QA rules, including chapter-concept alignment for Applications/Case Studies.
- Natural expert prose requirement.
- Duplicate and near-duplicate removal requirement.
- Consistent figure sizing / compact float-spacing requirement.
- All figures/tables/listings must be referenced and substantively discussed.
- All code lines must be numbered in the final manuscript.
- Full repository companion implementation required for every numbered listing.
- 19 Chapter 1 full listing implementations are committed by chapter/section.
- Chapter 1 listing smoke-test workflow passes.
- Cross-Domain Optimizer Selection Case Study rebuilt with reproducible figures/tables.
- Aerospace prognostics Application rebuilt with tri-format figure and benchmark table.
- Financial-fraud Application rebuilt from the public OpenML/ULB benchmark with tri-format figure and reproduced benchmark table.
- Chapter-specific ZIP builder and release workflow are committed.

## Remaining before Chapter 1 can be marked complete
- Reconstruct one integrated editable Chapter 1 `.tex` from the Pass20-authoritative content plus accepted remediations.
- Remove reconstruction artifacts, editorial/process prose, duplicate/similar content, and sections moved to later chapters.
- Complete all retained Applications/Case Studies under the updated chapter-alignment standard.
- Provenance-audit every remaining quantitative table.
- Individually audit every retained Chapter 1 figure for scientific validity, provenance, redundancy, caption fidelity, panel correctness, aesthetics, compact composition, and tri-format preservation.
- Ensure every retained figure/table/listing is referenced and substantively discussed in prose.
- Ensure Listing 1.9 and Listing 1.19 receive explicit line-range discussion after final code-line numbering is frozen.
- Compile the integrated chapter PDF.
- Verify final compiled length is 90-130 pages.
- Run final reference, sentence-completeness, Application/Case Study structure, duplicate-content, listing, figure-layout, and reproducibility QA from the same commit.
- Build `01_Foundations_of_Machine_Learning_PUBLICATION_READY.zip` only after all preceding gates pass.

## Required ZIP contents
- final editable Chapter 1 `.tex`
- compiled Chapter 1 PDF
- every retained figure in PNG, SVG, and PDF
- complete code-listing implementations organized by section
- listing and figure manifests
- provenance/checksum files
- package manifest with SHA-256 hashes

The ZIP must be committed under `book/releases/chapters/` and must correspond to the exact repository commit that passed final QA.
