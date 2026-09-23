# Code Listing Count and Quality QA

## Hard requirements

Every chapter must contain **15--22 substantive numbered code listings**.

A chapter fails QA if it contains:
- fewer than 15 listings;
- more than 22 listings;
- filler listings added only to satisfy the count;
- code dumps;
- untitled or unnumbered retained listings;
- listings not referenced and discussed in the prose;
- multiple listings that teach substantially the same implementation pattern;
- long setup, installation, configuration, serialization, or boilerplate blocks that belong in GitHub.

## KEEP / MERGE / MOVE / DELETE decision rule

For every code block or listing:

**KEEP** when it teaches a distinct chapter concept and is concise enough for the printed book.

**MERGE** when two or more listings teach closely related steps that are clearer as one coherent listing.

**MOVE TO GITHUB** when the code is useful for reproducibility but too long, repetitive, or implementation-heavy for the manuscript.

**DELETE** when the code is redundant, trivial, obsolete, or not discussed in the text.

## Preferred listing characteristics

- 15--22 per chapter.
- Most listings should be compact, typically under about 50 substantive lines unless the concept genuinely requires more.
- Each listing must have a descriptive caption/title and stable label.
- Each listing must be referenced and interpreted in nearby prose.
- Complete runnable implementations remain in GitHub.

## Release criteria

- Chapter listing count between 15 and 22 inclusive: PASS
- Code dumps: 0
- Filler listings: 0
- Untitled/unreferenced listings: 0
- Redundant listings: 0
- Interrupted multi-page listings: 0
