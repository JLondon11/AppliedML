# Generative AI Part II

## Canonical release contract
This chapter release must contain:
- `manuscript/chapter.tex`
- `manuscript/chapter.pdf`
- `figures/FIGURE_MANIFEST.csv`
- `figures/by_section/`
- `code/CODE_MANIFEST.csv`
- `code/REPRODUCIBILITY.md`
- `instructor/solutions.tex`
- `instructor/solutions.pdf`
- `qa/CHAPTER_QA.md`
- `qa/CODE_AUDIT.csv`

## Existing repository material
Current chapter-specific code, figure generators, experiment outputs, and provenance are under:
`chapters/generative_ai_part_ii`

## Certification rule
A script is not considered reproducible merely because it exists. Every executable artifact must be checked for syntax, dependency declaration, deterministic settings where applicable, input/data provenance, expected outputs, and correspondence to the manuscript or final figure it supports.
