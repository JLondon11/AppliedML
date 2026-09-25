# Pass 17 — Figure Aesthetic Audit

## Scope

This pass applies the book-wide aesthetic hard gate to every currently rendered figure asset available in the repository. Scientific validity and aesthetic quality are evaluated separately.

The canonical manuscript contains **415 figures**. At this stage the repository contains rendered PNG assets for only a small subset, concentrated in NLP Part II plus Scientific AI Figure 14. Therefore the statuses below are asset-level decisions only for figures that can actually be inspected. The remaining canonical figures are **BLOCKED-AESTHETIC** until their final render exists.

## Asset-level aesthetic decisions

### ACCEPT

- **NLP II Figure 4 — Embedding Geometry in Contextual Representation Spaces.**
  Balanced three-panel composition; muted semantic palette; clean centroid separation; quantitative heatmap with legible colorbar; final label-spacing corrections remove collisions.

- **NLP II Figure 6 — Functional Specialization of Transformer Attention Heads.**
  Coherent four-panel heatmap composition; single quantitative color scale; restrained terracotta peak markers; compact spacing; no infographic grammar.

- **NLP II Figure 7 — BERT Sentiment Computation as Measured Tensor States.**
  Strong three-panel hierarchy; diverging activation map, restrained navy/terracotta representation trajectories, and compact classifier output; reads as one scientific composition.

- **NLP II Figure 8 — IMDb Sentiment Confusion Matrix.**
  Clean quantitative matrix; counts plus row-normalized percentages; high legibility; colorbar remains dominant enough to communicate scale without decoration.

- **NLP II Figure 9 — BERT Fine-Tuning Dynamics.**
  Restrained navy training curve, terracotta validation curve, light-violet selected checkpoint; strong hierarchy and minimal clutter.

- **NLP II Figure 12 — Claim-Level Retrieval-Grounded Generation Comparison.**
  Muted paired scatter with identity line and compact uncertainty annotation; residual-failure panel uses a restrained slate/terracotta contrast; materially clearer than prior spaghetti-line design.

- **Scientific AI Figure 14 — Grad-CAM Retinal Methodological Demonstration.**
  Balanced real-image / activation / overlay composition; clean 0–1 colorbar; reduced overlay dominance; compact white-background scientific treatment.

### REVISE — AESTHETIC

- **NLP II Figure 3 — BPE Vocabulary Construction.**
  Scientific content is useful, but long rotated merge-operation labels dominate panel (a), the default blue/orange/green/red palette is visually generic, grid lines are unnecessary, and panel balance is weak. Replace long categorical strings with compact merge indices or a short merge table/annotation and apply the book palette.

- **NLP II Figure 5 — Layer-Wise Transformer Representations.**
  Scientifically strong and already quantitative, but panels (b) and (c) retain default blue-line styling and the overall visual language is less refined than the newly accepted figures. Normalize typography, palette, marker weight, and spacing while preserving the CKA colorbar.

- **NLP II Figure 10 — Enterprise RAG Quantitative Architecture.**
  Scientifically useful measured retrieval/reranking/evidence panels, but default blue bars plus default blue/orange/green lines make it look like stock plotting output. Retain the quantitative design and replace the defaults with a restrained semantic palette, tighter legend treatment, and more balanced panel widths.

- **NLP II Figure 11 — Retrieval Quality versus top-k.**
  Scientifically appropriate, but nearly all data are rendered in default blue and the dual-axis third panel lacks visual differentiation. Refine line/marker hierarchy, use restrained contrasting colors for cost versus relevance, and reduce the generic plotting appearance without obscuring the quantitative tradeoff.

## Obsolete / duplicate render artifact

The repository contains both `figure_03_bpe.png` and `figure_03_bpe_vocabulary_construction.png`. The latter is a stale/secondary artifact and should not be treated as an independent book figure. Only one canonical Figure 3 render should remain in the final release package.

## Book-wide conclusion

Rendered-asset aesthetic QA currently covers **11 unique figure concepts**:
- **7 ACCEPT**
- **4 REVISE-AESTHETIC**
- **404 BLOCKED-AESTHETIC** because their final rendered assets are not yet synchronized for inspection.

A caption or generation script is not sufficient for aesthetic acceptance. Final ACCEPT requires direct visual inspection at publication scale.

## Hard aesthetic release gate

A figure is REVISE if any of the following remain:
- default plotting-library palette or visual grammar;
- excessive whitespace or poor panel balance;
- label collisions or unreadable rotated text;
- legends/colorbars competing with the data;
- inconsistent typography or line weights;
- overly bright or visually crude color use;
- unnecessary grid lines or decoration;
- multi-panel artwork that appears assembled rather than designed;
- visual density that obscures the scientific comparison.

Scientific correctness remains mandatory; aesthetic refinement may never alter or embellish the underlying result.
