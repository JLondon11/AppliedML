# Editorial and Code-Listing Standard

The printed manuscript and the repository serve different purposes.

## Manuscript rules
- No editorial history, manuscript-development commentary, "original code" correction narratives, production-pass commentary, frozen-inventory commentary, or placeholder/legacy-screenshot discussion may appear in the book.
- Historical code-correction Tip boxes are prohibited.
- Every chapter must contain 15--25 substantive numbered code listings; 18--22 is the preferred target range.
- Every retained code listing must have a descriptive title/caption and stable label.
- Every retained code listing must be explicitly referenced and discussed in the manuscript.
- Code listings must be important and directly relevant to the chapter and teach a core algorithm, model component, evaluation method, experiment, data-processing method, systems pattern, or deployment workflow.
- Imports, downloads, setup boilerplate, duplicated variants, repetitive helper code, plotting-only code, figure-generation code, and long operational command dumps belong in the repository, not the printed chapter.
- No filler listings may be added merely to satisfy the minimum count.
- A retained listing should normally remain below about 50 substantive lines unless a longer listing is necessary for pedagogical completeness.
- No figure, table, sidebar, callout, algorithm float, or other floating object may interrupt a code listing.
- When a listing continues onto another page, its continuation must precede any figure or table that would otherwise float between listing segments.
- Multi-page listings must remain one contiguous listing object with continuous code order and stable caption/label identity.
- Figures and tables associated with a listing must be placed wholly before or wholly after the listing.
- Do not split one logical listing into multiple listing environments merely to accommodate a float.

## Current normalization requirement
The earlier Pass 21 balance (6--14 retained code units per chapter) no longer satisfies the book standard. Every chapter must be normalized to 15--25 substantive listings before release.

## Release gates
- Listing count per chapter: 15--25
- Untitled retained listings: 0
- Unreferenced retained listings: 0
- References to removed listings: 0
- Irrelevant/filler retained listings: 0
- Listings interrupted by figures/tables/floats: 0
- Historical/editorial Tip boxes: 0
- Manuscript-development narrative hits: 0
- Chapter compile failures: 0
- Undefined references/citations: 0

Complete executable code remains a chapter release requirement in GitHub even when only a concise excerpt is printed in the book.
