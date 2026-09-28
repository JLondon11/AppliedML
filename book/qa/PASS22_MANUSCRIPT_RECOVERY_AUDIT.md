# Pass 22 Manuscript Recovery Audit

Date: 2026-09-28

## Purpose

Determine whether the missing canonical LaTeX manuscripts and compiled chapter PDFs can be recovered from existing GitHub branches or from the repository states associated with the manuscript/code QA milestones.

## Branches inspected

The repository currently exposes:

- `main`
- `book-release-pass20`
- `genai-part1-repro`

The `book-release-pass20` branch contains a book/chapter release scaffold, including per-chapter `manuscript/`, `figures/`, `code/`, `instructor/`, and `qa/` directories. However, the manuscript directories contain README placeholders rather than chapter source.

## Historical Git states inspected

Recursive Git trees were checked at the following commits:

- `0518664f2ebff139e3654b529b37f956d874ebde` — “Record Pass 21 editorial and code-density audit”
- `2d9d54b8a38dfe3257861fc2a9a30638ef640290` — “Update editorial code standard for chapter listing requirements”
- `b9f3432bdec0f952f6281dd51910cc9f8e6446ec` — “Add 15-22 code listing quality gate”
- `476426cd4fa9a9914c8f2f28692575421561e34b` — head of `book-release-pass20`

### Recovery result

Each inspected tree contains:

- **0 `.tex` files**
- **0 `.pdf` files**

GitHub code search on the current indexed repository likewise returns no LaTeX manuscript files.

## Conclusion

The reader-facing chapter manuscripts and compiled PDFs referenced by historical QA records were not committed in the inspected canonical repository states. Therefore those manuscripts cannot be recovered from the tested Git objects.

Historical statements such as “the complete 12-chapter manuscript was audited” may describe work performed against external/local/chat-generated source, but that source is not preserved in the repository states checked here.

## Release implication

Historical manuscript QA is useful as process history but is **not reproducible evidence for the current release**.

The canonical release must be rebuilt from the actual latest chapter source files supplied from their surviving location (for example author-local files, project uploads, or other retained artifacts), then committed under each chapter's `manuscript/` directory together with bibliography/build metadata.

Once synchronized, all manuscript-dependent QA must be rerun, including:

- citation/reference audit;
- 15–22 versus 15–25 listing-governance reconciliation and actual listing count;
- figure insertion and prose-reference audit;
- caption/panel audit;
- undefined-reference/citation compile audit;
- code-listing continuity;
- case-study/application structure;
- sentence completeness and natural-prose review;
- final page-layout and whitespace audit.

## Status

| Recovery target | Status |
|---|---|
| Current main LaTeX | ABSENT |
| book-release-pass20 LaTeX | ABSENT |
| Pass 21 audit commit LaTeX | ABSENT |
| Later listing-policy commit LaTeX | ABSENT |
| Compiled PDFs in inspected states | ABSENT |
| Recoverable canonical manuscript from tested Git history | NOT FOUND |

**Disposition: HOLD — source synchronization required.**
