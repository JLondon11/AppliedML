# Pass 17 — Priority Figure Aesthetic Refinement

## Purpose

Pass 17 adds visual-design QA to the already validated scientific figures. Numerical results, datasets, models, and scientific interpretations were preserved. The pass changed only rendering, typography, palette, spacing, hierarchy, and workflow separation where needed.

## Accepted figures

### NLP Part II Figure 4 — Embedding Geometry
**ACCEPT.** The final asset uses a restrained navy/slate/teal/terracotta categorical palette, larger but controlled point markers, manually resolved label collisions, a compact contextual-bank panel, and a quantitative cosine-distance heatmap. The final render was visually inspected after the last label-spacing rerender.

### NLP Part II Figure 6 — Attention-Head Specialization
**ACCEPT.** The four aggregate routing-score matrices use one shared quantitative colorbar, restrained cividis scaling, terracotta maximum markers, consistent axes and typography, and balanced panel spacing. No panel/colorbar overlap remains.

### NLP Part II Figure 7 — BERT Sentiment Tensor-State Rendering
**ACCEPT.** The token-state heatmap, layer-wise [CLS] dynamics, and classifier output now share a consistent premium visual language. Navy is used for the primary state-norm trajectory, terracotta for the secondary cosine trajectory and positive-class emphasis, with compact white-background composition.

### NLP Part II Figure 8 — IMDb Confusion Matrix
**ACCEPT.** The matrix remains deliberately quantitative and restrained: counts and row-normalized percentages remain central, with a clean quantitative colorbar and publication-scale typography.

### NLP Part II Figure 9 — IMDb Training Dynamics
**ACCEPT.** Navy training trajectory, terracotta validation trajectory, and light-violet checkpoint marker produce a clear visual hierarchy without default plotting colors. The selected checkpoint remains visually explicit without decorative clutter.

### NLP Part II Figure 12 — SciFact Retrieval Grounding
**ACCEPT.** Muted teal claim-level scatter, neutral identity line, compact confidence-interval annotation, and slate/terracotta residual-failure bars produce a cleaner scientific composition while retaining the statistically cautious interpretation.

### Scientific AI Figure 14 — RetinaMNIST Grad-CAM
**ACCEPT.** The three image panels are compact and visually balanced; overlay opacity is restrained, and the quantitative 0–1 Grad-CAM colorbar remains prominent without competing with the retinal images.

## Production changes

- Added a book-wide **Figure aesthetic-quality hard gate** to `book/EDITORIAL_MANUSCRIPT_POLICY.md`.
- Added `book/qa/FIGURE_AESTHETIC_QA.md`.
- Separated rendering from expensive model/data computation for NLP II Figures 4, 6, and 12.
- Publication renderers use committed validated numerical artifacts, while original extraction/experiment scripts remain in the repository for complete reproducibility.
- Dedicated render workflows now own their final assets and rebase before push, reducing binary-asset races.

## Book-wide visual rule

A figure is not final ACCEPT merely because it is scientifically correct. It must also look intentionally designed at publication size: restrained premium palette, compact white-space use, coherent multi-panel hierarchy, legible typography, clean quantitative legends/colorbars, no label collisions, and no default-plot or infographic appearance.
