# Final Repository Preservation Status

Authoritative manuscript baseline: Pass20 PDF.

## Local recovered figure source
The Pass20 recovery workspace contains all 415 canonical figures in all three formats:
- PNG: 415
- SVG: 415
- PDF: 415

Per-chapter tri-format counts:
- Ch01 Foundations of Machine Learning: 32
- Ch02 Deep Learning Part I: 70
- Ch03 Deep Learning Part II: 38
- Ch04 Computer Vision Part I: 14
- Ch05 Computer Vision Part II: 56
- Ch06 NLP Part I: 23
- Ch07 NLP Part II: 26
- Ch08 Generative AI Part I: 35
- Ch09 Generative AI Part II: 11
- Ch10 Reinforcement Learning: 33
- Ch11 Scientific AI: 54
- Ch12 Systems Engineering and MLOps: 23

These recovered assets are preservation sources only until each figure passes the final scientific, provenance, redundancy, caption, heatmap/colorbar, panel, and aesthetic audits. Later accepted/remediated assets supersede older Pass20 artwork when validated.

## Repository final-tree status
Branch: `book-publication-recovery-final`

Final chapter-organized binaries are being stored under:
`book/final/<chapter>/figures/`

Repository-resident figure blobs are reused by exact SHA whenever available so binaries are not recompressed or altered. A GitHub Actions workflow generates missing PDF/PNG siblings from committed SVGs and commits them back to the same branch.

No chapter is considered publication-ready until:
1. editable final .tex is committed;
2. every retained figure has PNG/SVG/PDF committed;
3. removed/replaced figures have all manuscript references updated;
4. tables, listings, Applications, Case Studies, references, and formatting pass final QA;
5. compiled chapter length is 90-130 pages.
