# Editorial Manuscript Cleanliness Policy

This policy is mandatory for all book chapters.

## Reader-facing manuscript content

Do not include:
- discussion of an "original chapter", "original code", "previous version", prior implementation, migration, reconstruction, production pass, frozen inventory, or internal QA;
- editorial comments, authoring history, placeholder/replacement history, or notes about how a figure or section was produced;
- internal provenance language that belongs in repository QA metadata rather than the textbook;
- code-compendium sections or implementation dumps.

Scientific caveats remain required when they materially define the evidence (for example: synthetic experiment, controlled numerical model, public dataset, published source, or simulator output).

## Code shown in the book

Every retained code listing must:
1. have a label;
2. be explicitly referenced and discussed in the prose;
3. teach a distinct concept not already demonstrated by another listing;
4. be concise enough to support exposition rather than replace it;
5. defer complete executable implementations, setup, serialization, boilerplate, and repetitive training loops to the repository.

The current production guideline caps normal manuscript listings at approximately 40 lines.

## Full code

Complete executable code remains mandatory in GitHub even when the manuscript shows only a focused excerpt.

## Pass 21 benchmark

The book-wide editorial cleanup reduced manuscript listings from 444 listings / 10,502 listing lines to 164 listings / 2,916 listing lines.

Deep Learning Part I was reduced from 176 listings / 3,154 listing lines to 5 focused listings / 114 listing lines.

At the Pass 21 release gate:
- all retained listings are labeled and referenced;
- no retained listing exceeds 39 lines;
- all 12 chapters compile with zero fatal errors, overfull boxes, undefined references, or undefined citations;
- all 415 accepted figures remain integrated.
