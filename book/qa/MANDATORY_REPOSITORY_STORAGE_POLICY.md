# Mandatory Repository Storage Policy

Branch: `book-publication-recovery-final`

GitHub is the authoritative durable storage location for the book project.

## Hard rule

A chapter artifact is **not considered saved, accepted, complete, or publication-ready** merely because it exists in a ChatGPT runtime, uploaded attachment, generated workflow workspace, or local ZIP. It counts as preserved only after it exists in this GitHub repository and has a verifiable repository path plus Git commit/blob SHA.

For **each of the 12 chapters**, GitHub must contain:

1. the editable chapter `.tex` source;
2. the compiled chapter PDF;
3. every retained figure in PNG, SVG, and PDF;
4. all figure-generation code and provenance/data files needed for reproducibility;
5. the full runnable implementation for every numbered code listing, organized by chapter and section;
6. figure, table, listing, provenance, and checksum manifests;
7. the chapter-specific `PUBLICATION_READY.zip` after final QA;
8. the ZIP SHA-256 checksum.

## Recovery baseline

Before final editing is complete, recovered editable chapter sources are also preserved in GitHub under a recovery/baseline path. Final publication sources are promoted under `book/final/<chapter>/`. Recovery baselines are never silently overwritten; they remain available for forensic comparison.

## Required chapter paths

- `book/final/01_Foundations_of_Machine_Learning/`
- `book/final/02_Deep_Learning_Part_I/`
- `book/final/03_Deep_Learning_Part_II/`
- `book/final/04_Computer_Vision_Part_I/`
- `book/final/05_Computer_Vision_Part_II/`
- `book/final/06_Natural_Language_Processing_Part_I/`
- `book/final/07_Natural_Language_Processing_Part_II/`
- `book/final/08_Generative_AI_Part_I/`
- `book/final/09_Generative_AI_Part_II/`
- `book/final/10_Reinforcement_Learning/`
- `book/final/11_Scientific_AI/`
- `book/final/12_Systems_Engineering_and_MLOps/`

## Completion gate

No chapter may be declared complete until every required repository artifact above is present and the final package was generated from the same commit that passed updated Master QA.
