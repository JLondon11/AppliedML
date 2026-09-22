# Case Study and Application Prose Normalization QA

## Hard rule

Case studies and applications must use continuous scholarly prose. Bolded topic labels used as paragraph starters or pseudo-headings are prohibited.

## Prohibited constructions

Examples that fail QA include:

- **Problem:** explanatory prose
- **Dataset:** explanatory prose
- **Method:** explanatory prose
- **Model:** explanatory prose
- **Experimental Setup:** explanatory prose
- **Results:** explanatory prose
- **Evaluation:** explanatory prose
- **Limitations:** explanatory prose
- **Deployment:** explanatory prose
- **Lessons Learned:** explanatory prose
- **Business Impact:** explanatory prose

This rule applies whether the label and prose are on the same line or whether the bolded label appears alone on one line and the explanatory text begins on the next.

## Required editorial treatment

For each case study and application:

1. Preserve or improve the introductory narrative paragraphs.
2. Integrate the problem definition into normal prose.
3. Introduce the dataset/data source in complete sentences and paragraphs.
4. Explain methodology and experimental design as connected technical exposition.
5. Present results with quantitative evidence and interpretation.
6. Discuss limitations, failure modes, operational constraints, and implications in prose.
7. Use actual LaTeX subsection/subsubsection headings only when the material warrants a genuine structural section.
8. Avoid template-like repetition across chapters.
9. Ensure transitions connect the introductory context to the empirical/technical analysis.
10. Verify that no topic label remains as a bold pseudo-heading.

## Automated search patterns

When manuscript sources are available, audit for LaTeX and Markdown patterns including:

- \\textbf{Problem}
- \\textbf{Dataset}
- \\textbf{Method}
- \\textbf{Model}
- \\textbf{Results}
- \\textbf{Limitations}
- \\textbf{Evaluation}
- \\textbf{Deployment}
- **Problem**
- **Dataset**
- **Method**
- **Results**
- **Limitations**

Automated hits require contextual review because legitimate emphasis inside a sentence is not necessarily a defect. The prohibited form is a bolded topic word/phrase functioning as a label for the following paragraph.

## Release criteria

- Bolded topic-label paragraphs: 0
- Isolated pseudo-heading label lines: 0
- Template-like case-study/application sections requiring prose reconstruction: 0
- Unresolved case-study/application prose-format defects: 0

A chapter fails editorial QA if any case study or application violates these criteria.
