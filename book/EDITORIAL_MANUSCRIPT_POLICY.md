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


### Book-wide scientific figure optimality hard gate

Every figure in every chapter must receive an individual scientific-optimality review before chapter acceptance.

For each figure, the review must determine whether a materially better scientific representation can be used for the same pedagogical purpose. A figure must not be accepted merely because it is technically correct, attractive, reproducible, or consistent with the frozen inventory.

The reviewer must evaluate:
- whether the visual form is scientifically appropriate to the question being communicated;
- whether a more informative conventional scientific form would improve interpretation, such as a distribution, uncertainty interval, residual plot, calibration plot, confusion matrix, ablation plot, sensitivity curve, phase diagram, distance matrix, dimensionality-reduction plot, or other evidence-appropriate representation;
- whether the figure shows the underlying data, computation, uncertainty, variability, or mechanism at the appropriate level of detail;
- whether an arrow/box diagram or infographic-like rendering should instead be replaced by a quantitative or computational scientific rendering;
- whether empirical panels use real data or executable computation rather than invented values;
- whether the figure is redundant with another figure in the same chapter;
- whether the figure is balanced with the chapter's applications, case studies, theory, and experiments;
- whether the caption, panel structure, color encoding, heatmap scale, typography, and layout support scientific interpretation;
- whether the current figure can be improved without violating the frozen figure inventory or changing the intended scientific claim.

Each figure must receive one of four statuses:
- **ACCEPT** — no materially better scientific representation is warranted;
- **REVISE** — underlying evidence is appropriate but visual/statistical representation should be improved;
- **REPLACE** — a materially stronger scientific representation should replace the current figure;
- **NOT YET CERTIFIED** — the final figure asset or compiled chapter view is unavailable for review.

A chapter may not receive final figure ACCEPT status until **every figure in that chapter is individually certified ACCEPT after any required remediation and final rendered-output inspection**.

Required release condition: **0 REVISE, 0 REPLACE, and 0 NOT YET CERTIFIED figures in every chapter.**


## Case-study and application prose-format hard gate

Every case study and application must read as continuous scholarly exposition rather than as a sequence of bolded topic labels followed by prose.

Do not use inline paragraph constructions such as:
- **Problem:** followed by explanatory prose;
- **Dataset:** followed by explanatory prose;
- **Method:** followed by explanatory prose;
- **Model:** followed by explanatory prose;
- **Experimental Setup:** followed by explanatory prose;
- **Results:** followed by explanatory prose;
- **Limitations:** followed by explanatory prose;
- **Lessons Learned:** followed by explanatory prose;
- **Deployment:** followed by explanatory prose;
- **Evaluation:** followed by explanatory prose;
- **Business Impact:** followed by explanatory prose;
- or similar bolded topic words functioning as pseudo-headings inside the body text.

These labels must not appear as isolated bold terms on separate lines with their explanatory text immediately following.

Instead:
- integrate the topic naturally into complete paragraphs;
- use real LaTeX subsection/subsubsection headings only when a genuine structural subdivision is warranted;
- preserve several introductory paragraphs where appropriate, then transition into coherent narrative treatment of the problem, data, methodology, experimental design, results, limitations, and implications;
- maintain paragraph-level continuity and avoid checklist-like or template-like prose;
- vary sentence structure so the case study reads as authored technical analysis rather than a filled-in form;
- ensure every case study and application includes substantive quantitative evidence, methodological reasoning, interpretation, and limitations where appropriate.

Required release condition: **0 bolded inline topic-label paragraphs and 0 pseudo-heading label lines in case studies or applications.**

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
- figures without individual scientific-optimality certification: 0;
- figures with REVISE, REPLACE, or NOT YET CERTIFIED status: 0;
- bolded inline topic-label paragraphs in case studies/applications: 0;
- pseudo-heading label lines in case studies/applications: 0;
- confirmed incomplete prose sentences: 0;
- confirmed dangling-conjunction sentence endings: 0;
- undefined references/citations: 0;
- fatal compile errors: 0.
