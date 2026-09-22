# Case Study and Application Structure QA

## Hard rule

Every case study and application must use a two-stage structure:

1. **Several detailed introductory paragraphs** providing context, motivation, background, significance, and connection to the chapter.
2. A structured technical treatment using **bolded field labels**, with each label on its own line and the explanatory prose beneath it.

## Required introductory narrative

Before the first structured label, every case study or application must contain several substantive paragraphs that explain, as appropriate:

- the real-world or scientific context;
- why the problem matters;
- technical and operational motivation;
- prior or domain background needed to understand the application;
- why the selected dataset, system, experiment, or deployment setting is meaningful;
- how the case study/application connects to the chapter's methods and concepts.

The introduction must be genuinely informative. One short setup paragraph is insufficient.

## Required structured format

After the introduction, use bolded field labels such as:

**Problem**

The problem statement and technical objective are developed here in complete prose.

**Dataset**

The dataset, source, provenance, sampling, splits, preprocessing, and relevant limitations are described here.

**Method**

The modeling or algorithmic approach is explained here.

**Experimental Setup**

Training, evaluation protocol, baselines, hyperparameters, computational constraints, and reproducibility details are explained here.

**Results**

Quantitative results, uncertainty, comparisons, and interpretation are presented here.

**Limitations**

Failure modes, caveats, generalization limits, operational constraints, and unresolved issues are discussed here.

Other labels such as **Model**, **Evaluation**, **Deployment**, **Lessons Learned**, or **Implications** may be added when appropriate.

## Formatting requirements

- Each bolded field label must be on a separate line.
- The prose belonging to that label must begin below it.
- A label and its explanatory prose must not share the same line.
- Do not stack multiple labels on one line.
- Do not reduce the labeled content to fragments or bullet-like notes.
- Use complete paragraphs under the labels.
- Keep labels consistent in typographic treatment.
- The exact set of labels may vary with the application, but the structure must remain technically coherent.

## Automated audit patterns

When manuscript sources are available, flag constructions such as:

- \\textbf{Problem}: text on the same line
- \\textbf{Dataset}: text on the same line
- \\textbf{Method}: text on the same line
- \\textbf{Results}: text on the same line
- Markdown equivalents such as **Problem:** text

Also flag case-study/application sections where the first bolded field label appears without several substantive introductory paragraphs preceding it.

## Release criteria

- Case studies/applications with insufficient introductory background: 0
- Bolded labels sharing a line with explanatory prose: 0
- Structured labels not isolated on their own lines: 0
- Labeled sections containing only fragments or superficial text: 0
- Unresolved case-study/application structure defects: 0

A chapter fails editorial QA if any case study or application violates these criteria.
