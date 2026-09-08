# Figure 8 — Confusion Matrix for BERT Sentiment Classification

Frozen section: Case Study: Large-Scale Sentiment Analysis Using BERT
Subsection: Results and Error Analysis
LaTeX label: `fig:bert_sentiment_confusion`

This case-study figure is generated from a real executable IMDb sentiment
experiment. DistilBERT is fine-tuned on a deterministic 4,000-review sample of
the official IMDb training split and evaluated on a deterministic 2,000-review
sample drawn exclusively from the official held-out test split.

The plotted 2x2 matrix is computed directly from saved test predictions and
distinguishes TN, FP, FN, and TP. The full per-review predictions/probabilities,
matrix counts, model revision, dataset identity, seed, hyperparameters, accuracy,
and F1 are retained as provenance. No performance value is manually supplied.

The artwork contains only the scientific matrix, axes, class labels, cell counts,
and color scale: no title, caption, explanatory prose, decorative boxes, or
infographic grammar. Qualitative interpretation of ambiguous reviews, mixed
sentiment, sarcasm, and domain-dependent language belongs in the manuscript,
not inside the figure.
