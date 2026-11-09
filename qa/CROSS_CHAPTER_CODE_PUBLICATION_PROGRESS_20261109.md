# Code-publication progress — 2026-11-09

**Mandate:** Full executable source for every application and case study throughout the textbook must be committed to GitHub, organized by chapter and section. Scientific figures claiming experimental results must be regenerated from authentic data, not generic or invented curves.

## Changes committed in this pass

- `qa/discovered_code_units_20261109.json`: inventory of **54 identifiable code-bearing unit groups**, representing **105 committed Python/notebook/shell source files** across five chapter families. This is based on repository paths and does **not** represent the complete number of manuscript units.
- `tools/audit_discovered_book_code.py`: standard-library script that checks all discovered source/evidence paths against `git ls-files`, checks the curated `book_code_units.json` registry, and reports outstanding certification gaps.
- `.github/workflows/book-code-publication.yml`: runs the diagnostic path audit on chapter source or registry changes and uploads an audit report.
- The earlier `CODE_PUBLICATION_STANDARD.md`, `CODE_PUBLICATION_AUDIT_20261109.md`, `book_code_units.json`, and `tools/audit_book_code_publication.py` remain in effect.

## Findings and non-certification statement

The discovered code groups are distributed as follows:
- computer_vision: 9
- deep_learning: 7
- foundations_machine_learning: 4
- generative_ai_part_i: 14
- nlp_part_ii: 20

This inventory purposely avoids counting generic chapter figure generators as fully executed case studies. The larger repository includes many additional scientific and production source scripts outside these directory patterns.

In the curated Chapter 2 register, **13 units** are currently tracked, of which **9 are blocked by missing model-specific code or evidence**; **4 have supporting code/evidence files**, but **zero are marked fully certified**. The registry does not yet cover every application/case study in all chapters.

## Verification requirements still outstanding

1. Expand the curated manuscript-to-GitHub manifest to every named application and case study across all chapters. Compare source listings, dataset lineage, executed result files, figure-generation scripts, and manuscript figure references.
2. Commit any missing complete executable experiment scripts, tests, measured results and figures. Verify each file via a GitHub tree or content read, recording commit hashes.
3. Run exact named-dataset experiments for currently blocked units. A code inventory cannot replace actual executions on authentic datasets.
4. Review scientific quality and runtime correctness; neither the inventory scanner nor its CI workflow certifies model correctness or experimental reproducibility.
5. Do not approve final chapters until code, results, visual QA and manuscript integration all pass.

**Overall status: GitHub publication infrastructure expanded and committed; universal code-publication compliance remains not certified.**
