# Springer Nature Book Production Standard — AppliedML

This file operationalizes the current Springer Nature book-manuscript guidance for this project.

## Authoritative style policy
- Use the current Springer Nature **book/monograph** LaTeX template/macros where available, or a compatible standard LaTeX book-class structure for monographs.
- Do not use a journal article template for the book.
- Do not hard-code custom body fonts or arbitrary per-chapter font sizes. The Springer book template/default hierarchy controls body text, headings, captions, lists, and other manuscript structures.
- Springer Nature creates the final production page layout; our manuscript must therefore be structurally clean, internally consistent, and template-compatible rather than manually imitating a journal layout.

## Required chapter contents
Every final chapter must contain, as applicable:
1. complete prose and mathematical development;
2. all retained scientific figures;
3. all quantitative tables;
4. all numbered code listings;
5. all equations;
6. worked examples;
7. Applications;
8. Case Studies;
9. scholarly references;
10. end-of-chapter problem set/exercises;
11. figure alt-text inventory;
12. full code and data/provenance links in the repository.

## Typography
- Use the Springer book template's font family and size hierarchy.
- Do not load custom body-font packages unless Springer explicitly requires them.
- Code is monospaced.
- Mathematical variables use standard LaTeX math italics.
- Display equations are centered; equation numbers remain right aligned.
- Captions and headings follow one consistent template-controlled hierarchy across all chapters.
- PDFs must embed all fonts.

## Figures
- Number by chapter (Figure 1.1, Figure 1.2, etc.).
- Cite each figure in sequential order.
- Caption text must remain in LaTeX, not embedded in artwork.
- Captions identify all panels and scientific encodings.
- Each figure must have a separate alt-text entry.
- Original artwork is stored in the repository. Project policy additionally requires PNG/SVG/PDF siblings for every retained figure.
- Distinguish data by more than color alone where feasible; use markers, patterns, direct labels, or line styles as appropriate.
- Third-party figure sources/permissions must be recorded when applicable.

## Tables
- Number by chapter (Table 1.1, Table 1.2, etc.).
- Every table must have a caption and be referenced in prose.
- Tables are native LaTeX tabular structures, not images.
- Quantitative tables require provenance or reproducible computation.
- Third-party table sources are included at the end of the caption when applicable.

## Exercises and problem sets
Every chapter ends with a substantive problem set. A typical graduate technical chapter should include a balanced mix of:
- conceptual questions;
- mathematical derivations/proofs;
- quantitative/computational exercises;
- engineering/application problems;
- open-ended research/design questions.

Problems must be chapter-specific and must test material actually taught in the chapter.

## Submission package
For each chapter, repository storage includes:
- final editable .tex;
- compiled chapter PDF with embedded fonts;
- original/required figure files;
- project-standard PNG/SVG/PDF figure siblings;
- tables and provenance;
- full code implementations;
- alt-text manifest;
- checksum/package manifest;
- PUBLICATION_READY.zip.

The repository copy and chapter ZIP must come from the same final passing commit.
