# Figure 5 — Layer-Wise Evolution of Transformer Representations

Frozen inventory section: **Interpretability, Mechanistic Understanding, and Internal Representations**  
Subsection: **Internal Representations in Transformer Architectures**  
LaTeX label: `fig:interpretability_layerwise_representations`

This is a true scientific rendering generated from actual hidden states of
`google-bert/bert-base-uncased`. It deliberately avoids assigning a rigid
single function to any layer.

Panels:
- **(a)** linear CKA across all embedding/transformer depths, measuring gradual
  redistribution of representational geometry;
- **(b)** same-word cross-context cosine similarity, quantifying retention and
  refinement of lexical identity across depth;
- **(c)** financial-versus-geographic `bank` separation relative to within-sense
  dispersion, quantifying contextual/semantic differentiation across depth.

The complete 768-dimensional target vectors, CKA matrix, layer-wise scalar
measurements, and exact model revision are saved as numerical provenance.
The artwork contains no embedded title/caption or infographic grammar.

Interpretive context: Tenney et al. (2019) found aggregate pipeline-like
localization of linguistic information in BERT, while later reanalysis cautions
against rigid one-function-per-layer interpretations. The present figure therefore
visualizes gradual measured redistribution rather than categorical layer labels.
