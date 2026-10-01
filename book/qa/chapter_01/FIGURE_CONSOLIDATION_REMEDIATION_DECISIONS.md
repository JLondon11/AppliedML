# Chapter 1 Figure Consolidation and Remediation Decisions

Authoritative manuscript baseline: Pass20 Chapter 1, reconciled with the updated Master QA.

The integrated Chapter 1 manuscript currently contains 26 inherited placeholder figure environments plus four newly accepted/remediated figures (optimizer benchmark, optimizer conditioning, fraud imbalance, and turbofan prognostics). This matrix applies the duplicate, chapter-alignment, and scientific-aesthetic gates before artwork is regenerated.

## Remove / move rather than regenerate

| Historical figure | Decision | Reason |
|---|---|---|
| Fig. 1.2 learning-paradigm taxonomy | REMOVE / merge into Fig. 1.1 | Substantially duplicates the paradigm overview. Keep one stronger taxonomy only. |
| Fig. 1.9 feature-engineering workflow | REMOVE / merge into Fig. 1.8 | Overlaps the broader data/representation workflow; retain one scientifically richer workflow. |
| Fig. 1.14 CI/CD / production deployment architecture | MOVE to Ch. 12 Systems Engineering & MLOps | Primary learning objective is deployment lifecycle engineering, not Foundations. |
| Fig. 1.16 SHAP/LIME/Integrated Gradients relationship diagram | REMOVE from Ch. 1 | Generic interpretability taxonomy and strongly overlaps later advanced chapters. |
| Fig. 1.21 multilayer neural-network architecture | MOVE to Ch. 2 Deep Learning Part I | Architecture fundamentals belong in Deep Learning. |
| Fig. 1.22 backpropagation computational graph | MOVE to Ch. 2 Deep Learning Part I | Backpropagation mechanism belongs in Deep Learning. |
| Fig. 1.23 overview of classical/deep/probabilistic/ensemble/RL methods | REMOVE | Semantically overlaps Fig. 1.1 and the Algorithm Landscape prose. |

These removals are intentional and must be accompanied by deletion/rewrite of in-text references. Historical numbering is not a reason to retain redundant figures.

## Retain but scientifically regenerate

| Historical figure | Scientific replacement |
|---|---|
| Fig. 1.3 hypothesis-space/model-capacity geometry | Reproducible model-capacity experiment showing underfit/intermediate/overfit polynomial hypotheses on the same sampled regression problem. |
| Fig. 1.4 inductive-bias necessity | Reproducible interpolation comparison: multiple hypotheses fit the same finite observations yet diverge between/away from samples. |
| Fig. 1.6 train/validation learning curves | Actual training dynamics from a reproducible classifier experiment, with distinct well-fit/underfit/overfit regimes derived from controlled capacity/regularization settings. |
| Fig. 1.7 bias-variance/generalization error | Monte Carlo polynomial-regression bias-variance decomposition with quantitative curves. |
| Fig. 1.10 dimensionality reduction | Public sklearn digits; PCA, t-SNE, spectral graph-manifold embedding. |
| Fig. 1.11 HPO method comparison | Actual optimization trajectories/best-so-far validation score under matched evaluation budgets for random, grid, and successive-halving-style search. |
| Fig. 1.12 ROC/PR/calibration/confusion metrics | Reproducible public-data classifier evaluation with actual ROC, PR, calibration, and confusion-matrix panels. |
| Fig. 1.13 cross-validation protocols | Quantitative split-assignment matrices for k-fold, stratified k-fold, and nested CV; no flowchart boxes/arrows. |
| Fig. 1.17 clustering comparison | Actual clustering outputs on matched public/synthetic benchmark data for k-means, hierarchical, DBSCAN, and GMM. |
| Fig. 1.18 support-vector classifier | Actual linear and kernel SVM decision surfaces/support vectors on a controlled benchmark dataset. |
| Fig. 1.19 decision-tree partitioning | Actual decision surface plus rendered tree-derived partition statistics. |
| Fig. 1.20 ensemble comparison | Actual bagging/boosting/stacking/random-forest decision surfaces or matched quantitative diagnostics; no generic ensemble cartoon. |
| Fig. 1.25 medical preprocessing | Public scikit-image immunohistochemistry sample with grayscale transformation and Otsu tissue mask. |

## Retain pending stronger evidence / domain-specific reproduction
- Fig. 1.1 consolidated paradigm taxonomy: retain only if redesigned as a compact scientific taxonomy without infographic grammar.
- Fig. 1.5 iterative ML engineering workflow: retain only if it remains compact and nonredundant with the data-quality workflow.
- Fig. 1.8 representation/data workflow: retain after merging Fig. 1.9 content.
- Fig. 1.15 responsible-AI governance: replace generic framework if possible with an evidence-driven fairness/calibration/robustness diagnostic; otherwise reduce to concise structured schematic.
- Fig. 1.24 predictive-maintenance application: must use real bearing/vibration data or an explicitly cited public benchmark; no invented diagnostic curves.
- Fig. 1.30 smart-grid forecasting: replace generic workflow with an evidence-based forecasting diagnostic from a public load dataset when the reproducible dataset artifact is available.

## Already accepted new figures
- optimizer selection benchmark — real public WDBC data; tri-format
- optimizer conditioning sensitivity — controlled reproducible computation; tri-format
- credit-card fraud imbalance benchmark — real OpenML/ULB data; tri-format
- turbofan prognostics — explicitly labeled numerical reconstruction plus published C-MAPSS table; tri-format

## Aesthetic gate for every regenerated figure
- white scientific background;
- compact composition with tight bounding box;
- canonical book figure width and similar visual height;
- restrained premium palette with variation across adjacent figures;
- no titles embedded in artwork;
- panel labels below panels;
- unified detailed caption in LaTeX;
- legends/colorbars only when they encode real quantitative meaning;
- no large empty margins or decorative infographic grammar.
