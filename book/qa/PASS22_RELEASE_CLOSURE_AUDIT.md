# Pass 22 Release-Closure Audit

Date: 2026-09-27

## Scope

This audit checks the current `main` branch of `JLondon11/AppliedML` against the repository's book-wide release and Master-QA rules. It focuses on reproducible release state, canonical manuscript availability, code-listing governance, figure scientific-optimality certification, and the NLP Part I citation-audit requirement.

## Executive disposition

**BOOK: HOLD — release closure incomplete.**

The repository contains substantial code/provenance and QA infrastructure, but the canonical reader-facing manuscript and complete final rendered figure sets are not synchronized into `main`. Consequently, book-wide manuscript, citation, listing, figure-placement, caption, and final visual QA cannot currently be reproduced from the canonical branch.

## 1. Canonical manuscript release blocker

`book/CHAPTER_RELEASE_STATUS.csv` marks `latex_source` and `pdf` as `NOT_YET_COMMITTED` for all 12 chapters.

The current recursive repository tree confirms that the required canonical `manuscript/` directories and chapter LaTeX sources are absent from the chapter roots on `main`.

### Consequence

The following cannot be certified from the current canonical repository:

- NLP Part I citation/reference density;
- actual reader-facing code-listing counts;
- listing labels and prose references;
- figure placement in the chapters;
- figure discussion in surrounding prose;
- caption-to-panel correspondence in the compiled manuscript;
- chapter cross-references and bibliography resolution;
- final page-level whitespace and float behavior;
- compiled chapter length;
- final manuscript prose quality.

Historical QA records that report successful manuscript compilation are therefore not independently reproducible from the present `main` state.

## 2. Code-listing governance conflict

Three repository records disagree:

1. `book/PASS21_EDITORIAL_CODE_AUDIT.md` reports final retained counts ranging from 3 to 31 listings.
2. `book/CODE_LISTING_NORMALIZATION_AUDIT.csv` reports a different set of retained code-unit counts ranging from 6 to 14.
3. `book/qa/CODE_LISTING_COUNT_QUALITY_QA.md`, committed 2026-09-23, defines the newer hard gate as **15–22 substantive numbered listings per chapter**.

The older `book/PASS21_EDITORIAL_CODE_STANDARD.md` uses 15–25, but it predates the 2026-09-23 15–22 hard gate.

### Audit disposition

- Historical Pass 21 listing counts: **STALE / NON-CANONICAL FOR RELEASE**
- Current listing-count compliance: **NOT YET CERTIFIABLE**
- Governance: **REVISE** — choose and state one authoritative range in all policy and audit files.
- Until explicitly reconciled, the newest repository hard gate is 15–22; however, the author has also stated a 15–25 target in the editorial workflow, so this conflict must be resolved deliberately rather than silently.

## 3. NLP Part I scholarly-reference audit

NLP Part I remains a priority because the author explicitly requested stronger citation/reference density.

Current `chapters/nlp_part_i` contains figure-generation and Pass-14 provenance code but no canonical manuscript source or bibliography.

### Audit disposition

**NLP Part I references: NOT YET CERTIFIED.**

Required closure procedure once the canonical manuscript is committed:

1. audit every section/subsection for foundational, landmark, modern, dataset/benchmark, and survey references;
2. verify all technical historical claims and benchmark claims have appropriate citations;
3. ensure equations/method descriptions that materially derive from named work cite the primary source;
4. confirm applications and case studies cite the dataset, protocol, baseline, and relevant domain literature;
5. remove orphan bibliography entries and resolve every citation key;
6. compile and verify zero undefined citations;
7. compare citation density against neighboring chapters so NLP Part I is not under-referenced.

## 4. Figure scientific-optimality status

`book/qa/BOOK_WIDE_FIGURE_OPTIMALITY_CERTIFICATION.md` correctly states that the full book is not yet certified because complete final figure assets are absent from the repository.

The detailed audit currently covers only a subset of NLP Part II plus Scientific AI Figure 14.

Source remediations are recorded for:
- NLP II Figures 3, 4, 6, 10, and 12;
- Scientific AI Figure 14.

But `book/qa/SCIENTIFIC_FIGURE_REMEDIATION_STATUS.md` explicitly marks those changes **SOURCE REMEDIATED / RENDER PENDING** until regenerated final assets are visually inspected.

### Audit disposition

No chapter may yet receive final figure ACCEPT unless every final figure asset is synchronized and individually inspected at publication scale.

## 5. Release-layout compliance

Every chapter release manifest requires:

- `manuscript/`
- `figures/`
- `code/`
- `instructor_solutions/`
- `qa/`

The current chapter roots remain largely historical/provenance trees. `book/CHAPTER_RELEASE_STATUS.csv` correspondingly reports figures as PARTIAL, instructor solutions as NOT_YET_COMMITTED, and QA as PARTIAL across the book.

### Audit disposition

All 12 chapters remain **HOLD** at the release-package level.

## 6. Required next closure sequence

1. Synchronize the current canonical LaTeX source, bibliography, and build metadata into each chapter's `manuscript/`.
2. Synchronize all final accepted figure assets into each chapter's `figures/`, preserving source/provenance mappings.
3. Reconcile the listing-count rule globally (15–22 vs 15–25) and regenerate the count audit from the actual manuscript.
4. Run the NLP Part I citation-density audit against its real manuscript.
5. Regenerate and visually inspect every source-remediated figure.
6. Perform per-figure frozen-inventory × caption × manuscript-reference × final-render verification.
7. Synchronize instructor solutions.
8. Compile all chapters and run undefined-reference/citation, float continuity, caption/panel, sentence-completeness, case-study structure, and natural-prose QA.
9. Update `book/CHAPTER_RELEASE_STATUS.csv` only from reproducible canonical assets.
10. Mark a chapter COMPLETE only when all release gates pass simultaneously.

## Final Pass 22 status

| Area | Status |
|---|---|
| Canonical manuscript availability | HOLD |
| Reproducible chapter PDFs | HOLD |
| NLP Part I references | NOT YET CERTIFIED |
| Code-listing governance | REVISE |
| Actual listing-count compliance | NOT YET CERTIFIED |
| Figure-code provenance infrastructure | ACCEPT |
| Full final figure synchronization | HOLD |
| Book-wide individual figure optimality | NOT YET CERTIFIED |
| Instructor solutions | HOLD |
| Global release | HOLD |

The highest-priority blocker is no longer broad content generation. It is synchronization of the canonical manuscript and final assets into the repository so that the remaining QA can be executed reproducibly.
