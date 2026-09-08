# Figure 6 — Functional Specialization of Transformer Attention Heads

**Section:** Interpretability, Mechanistic Understanding, and Internal Representations  
**Subsection:** Attention Analysis  
**LaTeX label:** `fig:attention_interpretability`

This figure is generated from actual pretrained transformer attention weights.
It contains no manually fabricated heatmaps.

The four panels are selected by explicit routing diagnostics:
- **(a) positional:** BERT head with maximal average previous-token attention;
- **(b) syntactic:** BERT head with maximal routing from the main verb
  `analyzed` to its grammatical subject `scientist` in a controlled sentence;
- **(c) semantic:** BERT head with maximal routing from `diagnosis` to
  `patient` in a controlled semantic context;
- **(d) induction-like:** GPT-2 head maximizing repeated-pattern routing from
  the second `B` to the token following the first `B` in `A B C A B`.

Attention intensity is shown directly from model outputs. These diagnostics are
descriptive routing patterns, **not causal explanations** of model behavior.

The repository stores each selected full attention matrix, all head-selection
score tables, selected layer/head IDs, exact model revisions, stimuli, native SVG,
300-DPI PNG, and executable generation code.
