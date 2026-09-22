# Figure 6 — Functional Specialization of Transformer Attention Heads

**Section:** Interpretability, Mechanistic Understanding, and Internal Representations  
**Subsection:** Attention Analysis  
**LaTeX label:** `fig:attention_interpretability`

This remediated scientific figure is generated from actual pretrained-model attention weights and measures routing specialization across **multiple controlled stimuli per diagnostic**, rather than selecting a head from a single sentence.

The four panels are layer-by-head mean routing-score maps:
- **(a) positional:** mean previous-token attention across multiple controlled BERT sentences;
- **(b) syntactic:** mean routing from a main verb to its grammatical subject across multiple controlled active sentences;
- **(c) semantic:** mean routing between semantically associated target nouns across multiple controlled contexts;
- **(d) induction-like:** mean GPT-2 repeated-pattern routing across multiple repeated-token prompts.

Each panel shows every layer/head combination. The marked cell is the maximum mean routing score for that diagnostic. A shared quantitative colorbar reports the mean attention routing score.

These measurements are descriptive routing diagnostics, **not causal explanations** of model behavior. The repository stores the complete layer-by-head score matrices, head rankings, exact model revisions, stimuli, native SVG, 300-DPI PNG, and executable generation code.
