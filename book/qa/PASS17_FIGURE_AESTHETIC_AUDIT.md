# Pass 17 — Figure Aesthetic Audit

## Hard standard

Scientific validity is necessary but not sufficient. Final ACCEPT requires direct inspection of the rendered asset at publication scale for composition, typography, whitespace, palette discipline, visual hierarchy, legends/colorbars, panel alignment, and print robustness.

## Priority figures closed

The following previously remediated figures now pass both scientific and aesthetic QA:

- NLP II Figure 4 — ACCEPT
  - compact three-panel composition;
  - restrained navy / teal / terracotta palette;
  - readable labels after collision cleanup;
  - quantitative cosine-distance heatmap and colorbar;
  - no default-plot appearance.

- NLP II Figure 6 — ACCEPT
  - four aligned diagnostic heatmaps;
  - consistent axes and panel geometry;
  - external quantitative colorbar;
  - restrained terracotta maxima markers;
  - no panel overlap or clutter.

- NLP II Figure 7 — ACCEPT
  - measured tensor-state heatmap, layer dynamics, and classifier output;
  - coherent navy / terracotta visual hierarchy;
  - quantitative activation colorbar;
  - no infographic arrow-chain grammar.

- NLP II Figure 8 — ACCEPT
  - quantitative confusion matrix;
  - raw counts plus row-normalized percentages;
  - legible colorbar and annotations;
  - compact balanced composition.

- NLP II Figure 9 — ACCEPT
  - navy training curve, terracotta validation trajectory, light-violet selected checkpoint;
  - clear hierarchy and restrained styling;
  - no unnecessary grid or default color cycle.

- NLP II Figure 12 — ACCEPT
  - claim-level paired scatter with identity line;
  - compact CI annotation;
  - residual-failure panel with restrained slate / terracotta bars;
  - bootstrap uncertainty retained without clutter.

- Scientific AI Figure 14 — ACCEPT
  - balanced three-panel Grad-CAM composition;
  - clean quantitative 0–1 colorbar;
  - less visually dominant overlay;
  - appropriate methodological rather than clinical framing.

## Book-wide next priorities

The manuscript retains 415 canonical figure environments. The next aesthetic audit should prioritize:

1. the 173 architecture / taxonomy / pipeline / workflow figures previously flagged for possible overuse of schematic grammar;
2. known duplicate-role groups in Deep Learning I, Computer Vision II, and NLP II;
3. all remaining heatmap / saliency / matrix figures for quantitative scale placement and visual balance;
4. figures with generic plotting-library color cycles, excessive whitespace, label collisions, or inconsistent panel sizes;
5. figures with dense multi-panel layouts that remain difficult to read at actual LaTeX scale.

## Release rule

A figure remains REVISE even when scientifically correct if it is visually crude, cluttered, overly bright, repetitive, typographically inconsistent, or obviously based on plotting-library defaults.

Final book-wide aesthetic certification remains incomplete until the rendered assets for all 415 figures are available and directly inspected.
