# Scientific Figure Optimality Audit

## Scope

This audit evaluates every final scientific figure asset currently present in the repository tree. The repository currently contains final artwork for NLP Part II Figures 1--12 and Scientific AI Figure 14. The remaining book figures are not present as final image assets in this repository snapshot and therefore cannot be visually/scientifically certified here.

The standard is stricter than "technically correct": a figure must be the strongest scientific representation reasonably available for its stated pedagogical purpose, while respecting provenance and the frozen inventory.

## Decision vocabulary

- **ACCEPT** — scientifically appropriate and already close to the strongest representation for its purpose.
- **REVISE** — underlying experiment/data are sound, but the visual/statistical representation should be improved.
- **REPLACE** — the current figure form is not the strongest scientific representation; redesign is warranted even if the underlying computation is valid.

## Figure-by-figure audit

| Figure | Decision | Scientific assessment | Required action |
|---|---|---|---|
| NLP II Fig. 1 — Scaling Laws Across Model Size, Data, and Compute | ACCEPT | Published fitted laws are evaluated directly, with no fabricated observations. The three panels separate model-size, data-size, and compute-optimal behavior cleanly. | Preserve. Ensure final caption explicitly states that curves are published fitted laws/derived frontiers rather than raw observations and describes (a)--(c). |
| NLP II Fig. 2 — Compute-Optimal Scaling and the Chinchilla Principle | ACCEPT | Fixed-compute IsoFLOP geometry, predicted loss around the optimum, and compute-optimal N/D scaling form a coherent quantitative scientific figure. | Preserve. Caption must distinguish fitted-model predictions from empirical measurements and describe all markers/panels. |
| NLP II Fig. 3 — Byte-Pair Encoding | REVISE | The plots are reproducible and quantitative, but the current three time-series panels do not show the actual merge mechanism as directly as they could. | Retain quantitative provenance but redesign one panel to expose the actual merge sequence/token-state evolution while keeping at least one quantitative compression/sequence-length panel. Avoid infographic grammar. |
| NLP II Fig. 4 — Embedding Geometry in Contextual Representation Spaces | REVISE | Actual BERT contextual vectors and PCA are sound, but the figure can overstate semantic structure when PCA axes omit explained-variance percentages and centroid arrows suggest relationships more strongly than the data justify. | Put explained variance on PC-axis labels; replace or strengthen panel (c) with a quantitative distance/separation representation (e.g., cosine-distance matrix or centroid separation with uncertainty) rather than interpretive arrows alone. |
| NLP II Fig. 5 — Layer-Wise Evolution of Transformer Representations | ACCEPT | CKA heatmap plus two independently computed layer-wise diagnostics is a strong scientific decomposition. The heatmap already has a quantitative CKA colorbar. | Preserve. Caption must explain the CKA scale and both scalar diagnostics. |
| NLP II Fig. 6 — Functional Specialization of Transformer Attention Heads | REPLACE | Heatmaps are derived from real attention, and a quantitative attention-weight colorbar is present, but selecting a "best" head from one controlled stimulus per claimed function is too fragile for a figure titled functional specialization. | Replace with an aggregate diagnostic over a larger stimulus set: head-by-head specialization scores with uncertainty/distribution across examples, with one or two exemplar attention matrices only as supporting panels. Preserve the caveat that attention is not causal explanation. |
| NLP II Fig. 7 — BERT Sentiment Classification Architecture | ACCEPT | Tensor/computation rendering is appropriate for an architecture figure, and it is grounded in real BERT configuration/provenance rather than invented performance. | Preserve if final visual remains compact and non-infographic. Caption should explain tensor flow, pretrained vs adapted components, and dimensions. |
| NLP II Fig. 8 — Confusion Matrix for BERT Sentiment Classification | ACCEPT | Real held-out predictions, direct TN/FP/FN/TP counts, and a quantitative colorbar make this the correct scientific representation. | Preserve. Consider including both count and row-normalized percentage in cell annotations only if legibility remains high. |
| NLP II Fig. 9 — Training and Validation Dynamics for BERT Fine-Tuning | ACCEPT | Measured training/validation loss versus optimization step with best-validation checkpoint is the appropriate scientific diagnostic. | Preserve. Caption should identify sampling protocol, checkpoint rule, and interpretation of divergence/overfitting. |
| NLP II Fig. 10 — Enterprise Retrieval-Augmented Generation Architecture | REVISE | The computation is real and dimensions/provenance are meaningful, but the rendering still relies heavily on an arrow-based pipeline, which is weaker scientifically than a data-state view. | Redesign toward measured computational states: query/document representation, similarity/ranking state, retained evidence IDs, prompt composition, and response grounding. Minimize generic arrow/flowchart grammar. |
| NLP II Fig. 11 — Retrieval Quality as a Function of Top-k Depth | ACCEPT | Real BEIR SciFact retrieval results, uncertainty, marginal gain, context burden, and irrelevant fraction directly expose the top-k tradeoff. | Preserve. Ensure panel (c)'s dual axes are visually unambiguous and caption explains both scales. |
| NLP II Fig. 12 — Hallucination Reduction Through Retrieval Grounding | REPLACE | The controlled experiment is reproducible, but two bar charts compress claim-level variability and the unsupported-term metric is only a lexical proxy. The current visual can imply more certainty than the protocol warrants. | Replace panel (a) with paired claim-level distribution/ECDF or paired difference visualization plus bootstrap interval for the mean/median change. Keep a clearly labeled residual-failure decomposition. Caption must foreground that lexical support is a proxy, not factuality adjudication. |
| Scientific AI Fig. 14 — Grad-CAM Explainability Pipeline for Retinal Classification | REVISE | Real held-out RetinaMNIST data and computed Grad-CAM are scientifically grounded, but the current 28x28 source resolution is weak for a clinical-audit figure and the heatmap lacks the newly required quantitative colorbar. | Add a 0--1 Grad-CAM colorbar immediately. Prefer a higher-resolution real retinal source/model experiment where feasible; otherwise explicitly present RetinaMNIST as a low-resolution methodological demonstration rather than a clinical-grade attribution example. |

## Immediate priorities

1. **REPLACE** NLP II Figure 6 with aggregate multi-example attention specialization evidence.
2. **REPLACE** NLP II Figure 12 with claim-level paired/distributional evidence rather than only bars.
3. **REVISE** Scientific AI Figure 14 to include a quantitative heatmap colorbar and improve image resolution/clinical realism.
4. **REVISE** NLP II Figure 10 away from predominantly arrow-based architecture grammar.
5. **REVISE** NLP II Figures 3 and 4 to strengthen direct scientific interpretation.

## Book-wide acceptance rule

A figure should not receive ACCEPT merely because it is correct, attractive, or reproducible. It must also satisfy all of the following:

- the visual form is appropriate to the scientific question;
- empirical claims use real data or actual executable computation;
- uncertainty/distribution is shown when scientifically material;
- heatmaps include quantitative colorbars/legends;
- multi-panel labels are fully described in the caption;
- visual encodings are interpretable without decorative infographic grammar;
- no duplicate or substantively redundant figure exists in the chapter;
- no stronger conventional scientific representation would materially improve correctness, interpretability, or evidentiary value.

## Full-book limitation

The repository does not currently contain the final artwork for the complete book-wide figure set. A full-book per-figure visual audit therefore requires the final chapter PDFs or all final figure assets to be synchronized into the repository. Until then, figures absent from the repository remain **NOT YET CERTIFIED** under this optimality standard.
