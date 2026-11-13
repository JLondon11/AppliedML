# Chapter 6 — Generative AI Part I: canonical release manifest

## Canonical root
`chapters/generative_ai_part_i/`

## Required release organization
- `manuscript/` — current LaTeX source and build metadata
- `figures/` — accepted production figure assets, mapped to exact manuscript captions
- `code/` — complete section-specific experiment implementations
- `figure_generation/` — measured-data figure generation and associated audits
- `instructor_solutions/` — complete instructor solutions where applicable
- `qa/` and `quality_assurance/` — scientific, code, provenance, build and release checks

## Audited baseline versus current release
The archived Pass 44 Master QA records a 109-page chapter containing **48 figures, 17 tables, 11 numbered code listings, and 14 applications/case studies**. These are archived Pass 44 counts, **not an independently verified inventory of the latest compiled manuscript**. Do not substitute the former `35 figures` placeholder or treat it as the current production target.

The Pass 50 scientific report and `qa/CH6_SCOPED_SCIENTIFIC_VS_PRODUCTION_REGISTER_20261113.csv` establish four separately scoped, original-data scientific certificates. This does not certify all fourteen parent applications; full-chapter **production approval remains HOLD** until the current exact LaTeX/PDF, figures, data-derived outputs, editorial checks, and all required code/asset relationships are inspected.

## Migration and code-delivery rule
Existing historical `figure_generation/`, `code/`, `qa/`, section-specific, and production-pass directories remain valid evidence sources. Index them into the canonical release layout rather than deleting them. All new and revised code must be committed to this repository and organized under the correct application or chapter-specific QA path. External licensed datasets and laboratory records may be referred to by immutable checksums and acquisition instructions rather than improperly rehosted.

## Completion gate
Only mark Chapter 6 COMPLETE after (1) all specifically claimed application scopes have scientific evidence passing proportional R1–R10, (2) every published figure and table is mapped to actual source-derived outputs, (3) repository code and code checks are verified, and (4) the current manuscript PDF/LaTeX passes production QA.
