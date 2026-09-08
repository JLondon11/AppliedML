# Figure 9 — Training and Validation Dynamics for BERT Fine-Tuning

Section: Case Study: Large-Scale Sentiment Analysis Using BERT
Subsection: Results and Error Analysis
LaTeX label: `fig:bert_sentiment_loss_curves`

This case-study figure is generated from a real executable IMDb fine-tuning
experiment. A pretrained DistilBERT sequence classifier is fine-tuned for three
epochs on a deterministic 4,000-review sample of the official IMDb training
split. Validation loss is measured repeatedly on a deterministic 2,000-review
sample from the official held-out test split.

The scientific plot contains measured training and validation cross-entropy loss
versus optimization step. A restrained vertical marker identifies the checkpoint
with minimum measured validation loss, making convergence, instability,
overfitting/divergence, and checkpoint-selection behavior directly inspectable.
No loss value is manually supplied.

The complete Trainer log history, separated training/validation loss tables,
model revision, seed, learning rate, batch sizes, epoch count, best checkpoint,
native SVG, 300-DPI PNG, and executable source are retained as provenance.
