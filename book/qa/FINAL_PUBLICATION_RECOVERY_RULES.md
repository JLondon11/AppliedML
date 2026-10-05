# AppliedML Final Publication Recovery Rules

Authoritative manuscript baseline: `AppliedML_Complete_Book_Proof_Pass20.pdf`. Pass20 is the latest complete book manuscript version and is the text source that must be updated line by line.
Later Pass15/Pass16 packages, accepted figure ZIPs, QA reports, and Book Editing chat history are superseding correction sources only; they do not replace Pass20 as the manuscript baseline.

## Manuscript hard gates
- Reconstruct editable LaTeX chapter sources; the final deliverable may not be a PDF-inclusion wrapper.
- Audit every chapter line by line for incomplete sentences, fragments, duplicated prose, broken cross-references, malformed headings, and extraction artifacts.
- Prose must read like natural expert technical writing rather than a repeated template: vary sentence length and syntax, use specific transitions, avoid canned framing and repetitive paragraph openings, remove generic filler, and preserve a consistent authorial voice across the book. Do not write to evade AI-detection systems; edit for clarity, specificity, coherence, and natural scholarly tone.
- Remove all editorial/production narrative, including discussion of original code/chapter drafts, repair history, placeholders, production passes, and instructions to future editors.
- Remove irrelevant Tips/Notes/tooltips about original code, missing code, prior chapter versions, or editorial process.
- Application and Case Study sections begin with multiple substantive introductory/background paragraphs. Only afterward use bold standalone labels such as **Problem**, **Dataset**, **Model**, **Method**, **Evaluation**, **Results**, **Error Analysis**, **Limitations**, and **Deployment/Research Implications**, each followed by full prose.
- Every retained Application and every retained Case Study must include at least one scientifically appropriate, nonredundant figure and at least one evidence-based benchmark/comparison table tied directly to that section.
- Every Application and Case Study must explicitly apply concepts taught in the chapter in which it appears. Its problem formulation, model/method choices, equations, diagnostics, evaluation, and interpretation must connect to that chapter's core material. Sections that primarily belong to another chapter must be moved, rewritten, merged, or removed rather than retained for breadth alone.
- The figure/table requirement must never be satisfied with decorative artwork, generic diagrams, duplicated chapter-level assets, invented numbers, or irrelevant leaderboards.
- Figures and tables may be based on cited published evidence or reproducible executable analysis using the stated dataset/protocol.
- If an Application or Case Study cannot support a scientifically meaningful figure and comparison table without repetition or unsupported evidence, consolidate it with a stronger neighboring section or remove it.
- Quantitative claims/tables require real provenance or reproduced experiments. Remove unsupported illustrative numerical claims.
- Ensure current/SOTA topics required by the later project history are present and integrated without duplicating specialized later chapters.

## Code hard gates
- 15-22 substantive code listings per chapter (never exceed 25 without explicit approval).
- Every retained listing must be titled, numbered, referenced, and discussed in the surrounding prose.
- Every code listing must display line numbers for all code lines. Surrounding prose should cite specific line ranges when those lines implement a concept, safeguard, metric, data split, loss term, optimization step, or other point that materially supports the chapter explanation.
- Every figure, table, and code listing must be explicitly referenced and substantively discussed in the surrounding text. No orphaned floats/listings and no token mentions that merely say a figure/table exists; the prose must explain what evidence or implementation detail the reader should take from it.
- Remove repetitive, trivial, repair-oriented, or non-informative code dumps.
- Retained code shown in the manuscript may be pedagogically focused, but the full and complete runnable implementation for every numbered listing MUST be committed in GitHub.
- Full listing implementations must be organized by chapter and section, with a stable one-to-one mapping from manuscript listing number to repository file. Use filenames that preserve the listing number and descriptive title; do not pool unrelated listing code in generic dump files.
- Each full implementation must include required imports, data-loading/setup code, helper functions, configuration, deterministic seeds where applicable, and execution entry points needed to reproduce the listing's behavior. If external data are required, document the exact source and expected path/API rather than embedding unavailable assumptions.
- The manuscript excerpt and repository implementation must remain semantically synchronized. Any listing edit that changes behavior requires updating both the manuscript and the corresponding repository file.
- Remove references/tooltips to code that does not exist or does not run.

## Figure hard gates
- Audit every canonical figure individually for scientific correctness, provenance, caption fidelity, and aesthetics.
- Updated Master QA is the controlling acceptance standard for every chapter, section, figure, table, listing, caption, reference, and compiled page. Earlier ACCEPT/FROZEN/CANONICAL labels do not override a current failure.
- Every retained figure must pass two independent gates: (1) scientific validity/provenance and (2) publication aesthetics. A scientifically correct but visually weak figure is REVISE; an attractive but scientifically weak or unsupported figure is REPLACE.
- Figure acceptance requires the best defensible representation for the scientific claim: correct method/data, informative encoding, appropriate uncertainty/legends/colorbars, readable typography, compact composition, restrained premium palette, balanced panel geometry, and no unnecessary infographic grammar.
- Each chapter and each section must be reviewed in context after figures/tables/listings are integrated so the surrounding prose actually explains the evidence and the final compiled layout remains coherent.
- Scientifically correct but visually weak figures remain REVISE, not ACCEPT.
- Use real data/executable computation for empirical, benchmark, application, and case-study claims.
- No invented benchmark values or infographic substitutes for scientific renderings.
- White scientific background where appropriate; premium restrained palette; compact whitespace; consistent typography/line weights; clean legends/colorbars.
- Figure dimensions must be visually consistent across the book. Default target is approximately 0.88-0.94\\textwidth for single-row figures, with aspect ratios chosen to keep most figures within a common visual height envelope. Multi-panel figures should be composed to comparable total height rather than allowed to become unusually tall or short.
- Artwork must be exported with tight bounding boxes and minimal internal padding; remove empty canvas margins and unused panel space.
- LaTeX figure placement must avoid large vertical gaps: do not use manual positive \\vspace around figures, keep caption spacing compact and consistent, and normalize float parameters/placement so figures sit close to the surrounding discussion without crowding text.
- Every heatmap/matrix/saliency map that encodes magnitude must include a quantitative legend/colorbar.
- Every panel `(a)`, `(b)`, `(c)`, etc. must be explicitly described in the unified caption and referenced where appropriate in the prose.
- Detect and remove/replace exact or semantic duplicate figures. If a figure is removed, remove or rewrite every corresponding in-text reference.
- Remove duplicate and near-duplicate content across the book, including prose, figures, captions, tables, worked examples, Applications, Case Studies, and code listings. Similar items may coexist only when they answer clearly different scientific or pedagogical questions; that distinction must be explicit in the surrounding text and evidence.
- Duplicate detection is both intra-chapter and cross-chapter. When two items overlap substantially, retain the stronger scientifically defensible version, merge complementary material, or redesign one item to serve a distinct role. Do not preserve duplication merely to maintain historical numbering.
- Final surviving figure binaries must be archived in PNG, SVG, and PDF and committed to GitHub.

## Known authoritative later corrections
- NLP II Figures 4, 6, 7, 8, 9, 12 and Scientific AI Figure 14 use the later Pass 16 scientific remediations as the scientific baseline; aesthetic refinements must not change validated computations/results.
- Deep Learning I duplicate/overlap groups: 2.16/2.17, 2.41/2.42, 2.44/2.45 require scientifically distinct roles.
- Computer Vision II 5.37/5.38 require distinct scientific roles rather than duplicate architecture diagrams.
- NLP II 7.13/7.15 are distinct and should not be removed merely as visual duplicates.
- Chapter 1 page-33 tip about trying L-BFGS is removed as nonessential editorial/tip material.

## Persistence
- All accepted text and binary changes are committed to branch `book-publication-recovery-final`.
- Local deliverables carry SHA-256 checksums and a chapter/figure manifest.

## Chapter completion package
- Chapter completion requires a chapter-specific ZIP archive committed to the repository. The ZIP must contain the final editable chapter `.tex`, the compiled chapter PDF, every retained figure in PNG/SVG/PDF, the complete code-listing implementations organized by section, figure/listing manifests, and checksum/provenance files.
- The ZIP must preserve the final chapter directory structure rather than flattening unrelated assets. The archive filename must include the chapter number, chapter title, and a `PUBLICATION_READY` marker.
- A chapter is not marked complete until the archive is generated from the same repository commit that passed final Master QA, compilation, page-count, reference, figure, table, listing, duplicate-content, and layout checks.

## Repository completeness hard gate
- GitHub is the authoritative storage location for every final chapter artifact. No chapter file may exist only in a chat runtime, local workspace, temporary ZIP, or workflow artifact.
- For every chapter, the repository must contain: final editable `.tex`; compiled chapter PDF; every retained figure in PNG, SVG, and PDF; complete code-listing implementations organized by section; figure/listing/table manifests; provenance/checksum files; and the chapter `PUBLICATION_READY.zip`.
- A chapter is incomplete until all of those files are committed on `book-publication-recovery-final` and their repository paths are recorded in the chapter package manifest.

## Equation typography hard gate
- All displayed equations must be horizontally centered in the text block. Equation numbers remain right aligned in the standard book style.
- The final manuscript must not use the LaTeX `fleqn` option, left-shifted display math, manual negative horizontal spacing, or section-specific equation indentation that defeats the centered equation style.
- Multi-line `align`, `aligned`, `gather`, and related displays must be visually balanced and centered as a display block while preserving mathematical alignment at relation symbols where appropriate.
- Inline mathematics remains inline; only display mathematics is subject to the centered-display rule.
