# Natural Authored-Prose QA

## Purpose

This QA standard verifies that the book reads as deliberate expert-authored technical prose rather than as repetitive, mechanical, or templated writing.

This is an editorial-quality requirement. It is not intended to optimize for or circumvent automated authorship detectors.

## Book-wide review dimensions

### 1. Sentence-level naturalness
Review for:
- excessive repetition of the same sentence length or syntax;
- recurring sentence openings;
- mechanical parallelism repeated across many paragraphs;
- incomplete or overcompressed sentences;
- unnecessarily ornate or generic wording;
- vague intensifiers or unsupported evaluative language.

### 2. Paragraph-level authorship
Every paragraph should have a clear technical purpose. Reject paragraphs that:
- restate the previous paragraph without adding information;
- use a generic topic sentence followed by generic elaboration;
- repeat a stock template used elsewhere;
- could be moved to another chapter with little or no modification;
- end with a predictable generic significance statement;
- contain only transition language and no substantive content.

### 3. Section-level variation
Sections should not repeatedly follow an identical rhetorical template. In particular, audit:
- chapter openings;
- subsection openings;
- application introductions;
- case-study introductions;
- results discussions;
- limitations sections;
- future directions;
- chapter conclusions.

The structure should reflect the technical subject rather than a fixed prose mold.

### 4. Transition quality
Flag excessive or mechanical repetition of phrases such as:
- Moreover,
- Furthermore,
- In addition,
- It is important to note,
- It should be noted,
- This section discusses,
- This subsection presents,
- In this context,
- From this perspective,
- Taken together,
- Overall,

These phrases are not banned individually. They fail QA when overused, repeated in close proximity, or used instead of a specific logical transition.

### 5. Cross-chapter duplication
Search for:
- exact duplicate paragraphs;
- near-duplicate paragraphs;
- repeated introductory formulas;
- repeated case-study framing language;
- repeated conclusion language;
- repeated figure/table introduction sentences;
- repeated problem/dataset/method/result wording across unrelated applications.

Substantive technical definitions may recur when necessary, but repeated prose should be rewritten unless exact consistency is pedagogically required.

### 6. Technical specificity
Prose should demonstrate subject-specific reasoning:
- concrete mechanisms;
- explicit assumptions;
- meaningful tradeoffs;
- limitations and failure modes;
- quantitative interpretation;
- comparisons tied to the actual method/data;
- connections to equations, figures, tables, and code.

Generic statements such as "this is important," "this improves performance," or "this demonstrates the effectiveness of the method" require specific supporting explanation.

## Automated heuristics

Automated scans may identify:
- repeated paragraph hashes;
- high-similarity paragraph pairs;
- repeated sentence openings;
- repeated transition phrases;
- unusually high recurrence of identical n-grams;
- clusters of paragraphs with highly similar length and syntax.

Automated flags are not final judgments. Every finding requires editorial review in context.

## Manual final-pass requirements

Each chapter must receive a read-through focused on:
1. whether the prose sounds natural when read aloud;
2. whether paragraph rhythm varies appropriately;
3. whether transitions are logically specific;
4. whether the chapter has a distinct technical narrative;
5. whether applications and case studies feel individually written;
6. whether conclusions synthesize rather than merely repeat;
7. whether any paragraph sounds generic, interchangeable, or mechanically produced.

## Release criteria

- Exact duplicate prose paragraphs: 0 unless explicitly justified
- Unresolved near-duplicate prose paragraphs: 0
- Repetitive stock-transition defects: 0
- Mechanically repeated paragraph templates: 0
- Generic interchangeable prose sections: 0
- Unresolved unnatural-cadence findings: 0

A chapter cannot receive final editorial ACCEPT until it passes both automated heuristics and a manual authored-prose read-through.
