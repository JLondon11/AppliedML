# Printed Code Policy

The Springer manuscript uses representative code only.

## Rule
A code listing is retained in print only when:
1. surrounding prose explicitly references the listing;
2. the listing teaches a distinct algorithmic or implementation concept;
3. it is not redundant setup, download, import, save, plotting, or repeated train/evaluate boilerplate;
4. the complete executable implementation is maintained in the chapter's GitHub code release.

## Pass 21 result
- Initial printed listings: 444
- Final printed listings: 150
- Deep Learning Part I: 176 -> 18
- NLP Part I: 71 -> 11
- All retained listings are explicitly referenced in prose.
- Editorial/code-repair narratives about earlier/original code are prohibited from the manuscript.

The repository code CI currently audits 68 committed Python files with 0 syntax failures and 0 reproducibility-review findings.
