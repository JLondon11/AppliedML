# Editorial Manuscript Cleanliness Policy

This policy is mandatory for all book chapters.

## Reader-facing manuscript content

Do not include:
- discussion of an "original chapter", "original code", "previous version", prior implementation, migration, reconstruction, production pass, frozen inventory, or internal QA;
- editorial comments, authoring history, placeholder/replacement history, or notes about how a figure or section was produced;
- internal provenance language that belongs in repository QA metadata rather than the textbook;
- code-compendium sections or implementation dumps.

Scientific caveats remain required when they materially define the evidence (for example: synthetic experiment, controlled numerical model, public dataset, published source, or simulator output).

## Sentence-completeness hard gate

Every reader-facing sentence must be grammatically complete.

Release QA must reject:
- sentences that terminate with a dangling conjunction or connector, including words such as `and`, `or`, `but`, `because`, `although`, `while`, `whereas`, `which`, `that`, `with`, `of`, `to`, `for`, or `by` when the construction is incomplete;
- truncated clauses caused by editing, extraction, or LaTeX reconstruction;
- sentence fragments presented as prose unless intentionally used as a heading, label, table entry, figure annotation, or list item;
- unmatched opening constructions whose dependent clause or complement is missing;
- abrupt paragraph endings that leave an unfinished grammatical dependency.

Automated scans are only a first-pass detector. Every flagged sentence must be reviewed in context, and every chapter must receive a final human-readable prose pass for completeness.

Required release condition: **0 confirmed incomplete sentences and 0 confirmed dangling-conjunction endings.**

## Figure-caption hard gate

Every figure caption must be detailed enough to explain what the reader is seeing, what the important visual encodings mean, and how the figure relates to the surrounding discussion.

For multi-panel figures:
- every visible panel label, including `(a)`, `(b)`, `(c)`, and any additional labels, must be explicitly referenced in the main caption;
- the caption must describe the content, purpose, and interpretation of each panel in panel order;
- no panel may be present without a corresponding caption description;
- panel descriptions must be integrated into the main caption rather than omitted or left implicit;
- when panels use different data, methods, conditions, axes, metrics, or visual encodings, those differences must be stated clearly;
- captions must define symbols, colors, line styles, error bars, uncertainty bands, markers, and other non-obvious encodings when needed for interpretation;
- captions for empirical figures must state the relevant dataset, experimental condition, or provenance when required to understand the result;
- captions should be sufficiently self-contained that a reader can understand the figure without searching the body text for basic interpretation.

Required release condition: **0 unlabeled/undescribed panels and 0 materially under-detailed figure captions.**

### Heatmap legend hard gate

Every heatmap, matrix visualization, correlation map, attention map, confusion-matrix heatmap, saliency heatmap, activation map, or other color-encoded scalar field must include a visible color scale or legend that states what numerical values the colors represent.

- The colorbar/legend must show the quantitative mapping between color and value.
- The quantity represented must be named, including units when applicable.
- Tick values or a clearly interpretable numeric range must be shown.
- A heatmap without an interpretable color scale fails QA, even if the caption describes the palette.

Required release condition: **0 heatmaps without a quantitative color legend/colorbar.**

### Duplicate-figure hard gate

No chapter may contain duplicate figures.

- Exact duplicate image assets within the same chapter are prohibited.
- Near-duplicate figures that communicate substantially the same visual information are also prohibited unless there is a clear, documented pedagogical distinction.
- Different filenames, formats, crops, or minor cosmetic changes do not make duplicated content acceptable.
- Final chapter QA must compare the figure inventory and compiled chapter visually to confirm uniqueness.

Required release condition: **0 exact duplicate figures and 0 unresolved near-duplicate figures within any chapter.**

## Code shown in the book

Every retained code listing must:
1. have a descriptive title/caption and stable label;
2. be explicitly referenced and discussed in the prose;
3. teach a distinct concept not already demonstrated by another listing;
4. be important and directly relevant to the chapter;
5. be concise enough to support exposition rather than replace it;
6. defer complete executable implementations, setup, serialization, boilerplate, and repetitive training loops to the repository.

Every chapter must contain 15--25 substantive numbered code listings; 18--22 is the preferred target range. No filler listing may be introduced merely to satisfy the minimum.

### Listing continuity and float-placement hard gate

A code listing is a single contiguous reader-facing object.

- No figure, table, sidebar, callout, algorithm float, or other floating object may be inserted between two portions of the same listing.
- If a listing extends across a page boundary, the continuation must appear immediately on the next page before any intervening figure or table.
- A multi-page listing must preserve uninterrupted code order, line numbering, caption identity, and label identity.
- Figures and tables referenced near a listing must be placed either entirely before the listing begins or after the listing ends; they may not float into the listing's continuation area.
- A listing must not be manually split into separate listing environments merely to allow a figure or table to appear between them.
- If page composition creates a conflict, move the figure/table, move the complete listing, shorten the listing without losing pedagogical substance, or move supporting code to GitHub. Never interrupt the listing.
- Final PDF QA must visually inspect every multi-page listing to confirm that no float breaks its continuity.

Required release condition: **0 listings interrupted by figures, tables, or other floats.**

## Full code

Complete executable code remains mandatory in GitHub even when the manuscript shows only a focused excerpt.

## Release gates

A chapter may not be accepted until all of the following are true:
- 15--25 substantive numbered code listings;
- untitled retained listings: 0;
- unreferenced retained listings: 0;
- listings interrupted by figures/tables/floats: 0;
- unlabeled or undescribed figure panels: 0;
- materially under-detailed figure captions: 0;
- heatmaps without quantitative color legends/colorbars: 0;
- duplicate or unresolved near-duplicate figures within a chapter: 0;
- confirmed incomplete prose sentences: 0;
- confirmed dangling-conjunction sentence endings: 0;
- undefined references/citations: 0;
- fatal compile errors: 0.
