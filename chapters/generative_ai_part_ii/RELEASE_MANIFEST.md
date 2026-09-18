# 9. Generative AI Part II — Canonical Release Manifest

## Canonical root
chapters/generative_ai_part_ii

## Required release organization
- manuscript/ — LaTeX source and build metadata
- figures/ — 11 accepted production figures organized by section
- code/ — complete executable code
- instructor_solutions/ — complete instructor solutions
- qa/ — code, reproducibility, figure, compile, and release audits

## Current migration rule
Existing historical figure-generation, case-study, experiment, and production-pass directories remain valid provenance sources. They must be indexed into the canonical release directories rather than deleted.

## Completion gate
This chapter may be marked COMPLETE only after all five release areas above are present and the repository reproducibility workflow passes.