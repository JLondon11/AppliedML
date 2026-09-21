# Printed Code Policy

The Springer manuscript uses representative, pedagogically important code only.

## Hard chapter requirement
Every chapter must contain **15--25 substantive numbered code listings**.

A printed code listing is acceptable only when all of the following are true:
1. it has a clear, descriptive title;
2. surrounding prose explicitly references the listing by number and discusses why it matters;
3. the listing is important and directly relevant to the chapter's technical content;
4. it teaches a distinct algorithm, model component, evaluation method, experiment, data-processing method, systems pattern, or deployment workflow;
5. it is not redundant setup, download, import, save, plotting, repeated train/evaluate boilerplate, or trivial API syntax;
6. the complete executable implementation is maintained in the chapter's GitHub code release.

## Balance rule
- Minimum per chapter: 15
- Target range: 18--22
- Maximum per chapter: 25
- No filler code may be introduced solely to meet the minimum.
- When a chapter is below 15, add missing substantive computational examples that strengthen the pedagogy.
- When a chapter exceeds 25, merge related fragments or move supporting/reproducibility code to GitHub.

## Title and reference gate
Every retained numbered listing must have:
- a descriptive listing title/caption;
- a stable label;
- at least one explicit in-text reference;
- nearby prose explaining the listing's purpose, interpretation, or connection to the chapter.

Orphaned, untitled, unreferenced, or purely operational listings fail release QA.

## Repository boundary
Imports, environment setup, dataset-download utilities, plotting-only scripts, figure-generation code, helper functions, repeated training loops, long command sequences, and full reproduction packages normally belong in GitHub rather than the printed chapter.

This policy supersedes earlier Pass 21 listing-count guidance.
