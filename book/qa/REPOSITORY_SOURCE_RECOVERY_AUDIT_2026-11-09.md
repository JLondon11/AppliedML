# Repository Source Recovery Audit — 2026-11-09

## Scope and evidence

Read-only inventory of the `main` Git tree for `JLondon11/AppliedML` as returned by GitHub's recursive tree endpoint on 2026-11-09 (UTC date). This report inventories **committed files**, not older chat attachments, GitHub Actions artifacts, releases, ignored files, or local author manuscripts. No code was executed, and no experimental result was certified in this audit.

Tree entries: 643; committed files (blobs): 500. Tree response truncated: false.

## Canonical chapter release inventory

| Chapter | Topic | Committed files | Files in manuscript/ | All .tex | All .pdf | Files in figures/ | Files in instructor_solutions/ | Release manifest |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | Foundations of Machine Learning | 81 | 0 | 0 | 22 | 67 | 0 | yes |
| 2 | Deep Learning Part I | 18 | 0 | 2 | 0 | 0 | 0 | yes |
| 3 | Deep Learning Part II | 2 | 0 | 0 | 0 | 0 | 0 | yes |
| 4 | Computer Vision Part I | 5 | 0 | 0 | 0 | 0 | 0 | yes |
| 5 | Computer Vision Part II | 94 | 0 | 0 | 0 | 0 | 0 | yes |
| 6 | NLP Part I | 6 | 0 | 0 | 0 | 1 | 0 | yes |
| 7 | NLP Part II | 127 | 0 | 0 | 0 | 0 | 0 | yes |
| 8 | Generative AI Part I | 50 | 0 | 0 | 0 | 0 | 0 | yes |
| 9 | Generative AI Part II | 4 | 0 | 0 | 0 | 0 | 0 | yes |
| 10 | Reinforcement Learning | 2 | 0 | 0 | 0 | 0 | 0 | yes |
| 11 | Scientific AI | 30 | 0 | 0 | 0 | 0 | 0 | yes |
| 12 | Systems Engineering and MLOps | 2 | 0 | 0 | 0 | 0 | 0 | yes |

**Meaning of counts:** `figures/` is the canonical literal directory under each chapter; historic generator and experimental folders are not counted as `figures/`. The counts are not measures of scientific quality. PDF counts include figure PDFs and do not imply manuscript PDF availability.

## Critical findings

1. No chapter has a committed file in its canonical `manuscript/` directory. There is no complete manuscript source in this tracked Git tree.
2. Only two LaTeX files exist anywhere in the repository; both are experiment-specific Chapter 2 snippets:
   - `chapters/deep_learning/part_i/experiments/digits_threeway_cpu/manuscript_integration.tex`
   - `chapters/deep_learning/part_i/experiments/gradient_flow_multiseed_digits/manuscript_results.tex`
3. All 22 committed PDF files are Chapter 1 figure assets under `chapters/foundations_machine_learning/figures/Chapter_1_013/`; none is an identified complete chapter PDF.
4. No chapter has a committed file under its canonical `instructor_solutions/` release directory.
5. All twelve chapters have a `RELEASE_MANIFEST.md` file. Release manifests describe completion gates; their presence alone is **not** evidence that the gates passed.
6. The book-wide figure audit at `book/qa/BOOK_WIDE_FIGURE_OPTIMALITY_CERTIFICATION.md` explicitly leaves every chapter not yet fully figure-certified.

## Recover and reconcile — blocking actions

1. Recover each chapter's latest actual .tex, bibliography, styles, generated figures, and compiled PDF from the prior production environment/archives; compare version identifiers and content hashes before selecting the authoritative source.
2. Place current manuscript and build dependencies in the canonical `manuscript/` tree; retain historic code and generator directories and index them rather than deleting them.
3. For each application and case study, link a real dataset and split protocol, exact executable model, experiment output, measured metrics, and manuscript figure/table. Missing executions or dataset access must remain explicit blockers.
4. Compile chapters and inspect rendered equations, captions, figure numbering, cross-references, tables, code listings, prose depth, and editorial artifacts.
5. Implement end-to-end release gates for completeness and for full figure/case-study experimental certification; never mark a chapter COMPLETE solely from the presence of source code or QA prose.

**Overall status: BLOCKED ON MANUSCRIPT SOURCE RECOVERY; NOT BOOK-WIDE CERTIFIED.**
