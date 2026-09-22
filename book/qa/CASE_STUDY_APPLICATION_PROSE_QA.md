# Case Study and Application Structure QA

## Hard rule

Every case study and application must use a two-stage structure:

1. **Several detailed introductory paragraphs** providing context, motivation, background, significance, and connection to the chapter.
2. Structured technical paragraphs that begin with a **bolded lead-in topic word or phrase**, followed immediately by the explanatory prose on the same line.

## Required introductory narrative

Before the first structured lead-in, every case study or application must contain several substantive paragraphs explaining, as appropriate:

- the real-world or scientific context;
- why the problem matters;
- technical and operational motivation;
- prior or domain background needed to understand the application;
- why the selected dataset, system, experiment, or deployment setting is meaningful;
- how the case study/application connects to the chapter's methods and concepts.

One short setup paragraph is insufficient.

## Required structured format

Use paragraphs such as:

**Problem.** The problem statement and technical objective are developed here in complete prose.

**Dataset.** The dataset, source, provenance, sampling, splits, preprocessing, and relevant limitations are described here.

**Method.** The modeling or algorithmic approach is explained here.

**Experimental Setup.** Training, evaluation protocol, baselines, hyperparameters, computational constraints, and reproducibility details are explained here.

**Results.** Quantitative results, uncertainty, comparisons, and interpretation are presented here.

**Limitations.** Failure modes, caveats, generalization limits, operational constraints, and unresolved issues are discussed here.

Other lead-ins such as **Model.**, **Evaluation.**, **Deployment.**, **Lessons Learned.**, or **Implications.** may be added where appropriate.

## Formatting requirements

- The bolded lead-in must start the paragraph.
- The prose belonging to that lead-in must continue immediately after it on the same line.
- Do not put a bolded keyword on a line by itself.
- Do not put the explanatory prose below an isolated keyword.
- Do not stack multiple lead-ins in a single paragraph.
- Use complete, substantive paragraphs rather than fragments.
- Keep lead-ins consistent in typographic treatment and punctuation.
- The exact set of lead-ins may vary with the application, but the technical structure must remain coherent.

## Automated audit patterns

When manuscript sources are available, flag:
- isolated lines containing only \\textbf{Problem.}, \\textbf{Dataset.}, \\textbf{Method.}, \\textbf{Results.}, etc.;
- a line break immediately after a bolded lead-in before explanatory prose;
- case-study/application sections where the first lead-in appears without several substantive introductory paragraphs preceding it.

## Release criteria

- Case studies/applications with insufficient introductory background: 0
- Isolated keyword-only lead-in lines: 0
- Labeled paragraphs whose prose does not continue on the same line: 0
- Labeled sections containing only fragments or superficial text: 0
- Unresolved case-study/application structure defects: 0

A chapter fails editorial QA if any case study or application violates these criteria.
