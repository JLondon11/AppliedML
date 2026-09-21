# Manuscript Sentence-Completeness QA

## Scope
Apply this audit to every chapter, caption, sidebar, callout, case study, application, exercise solution, and surrounding prose before publication.

## Hard rule
Reader-facing prose may not contain incomplete or accidentally truncated sentences.

## Mandatory checks
1. Flag sentence endings whose final lexical token is a likely dangling connector or function word, especially:
   - and, or, but, nor, yet, so
   - because, although, though, while, whereas, if, unless, since
   - which, that, who, whose, whom
   - with, without, of, to, for, from, by, into, onto, through
2. Inspect text immediately before section, subsection, figure, table, equation, listing, list, and page-break boundaries for truncation.
3. Inspect paragraphs modified during merging/reconstruction for missing dependent clauses, complements, objects, or predicates.
4. Exclude legitimate headings, labels, table cells, equation fragments, code, citations, and intentionally telegraphic list items from false positives.
5. Manually review every automated hit in context.

## Release criterion
- Confirmed incomplete sentences: 0
- Confirmed sentence endings with dangling conjunctions/connectors: 0
- Unresolved automated sentence-completeness flags: 0

A chapter that fails any criterion returns to editorial revision.
