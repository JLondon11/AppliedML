# Book scientific and aesthetic figure audit — 2026-11-09

## Mandatory publication standard
All empirical figure values, predictions, model outputs, confusion matrices and experimental plots must be generated from genuinely recorded datasets and actual executable code. The figure must accurately label the source, training/validation/test partition, units, measured quantities, model and comparison. Genuine photographs are real data. Actually generated outputs of a model trained on real data may be shown if clearly labeled as generated results. Hypothetical metrics, synthetic "observations", and invented comparisons are prohibited as empirical evidence. Conceptual diagrams are permitted only as explicitly identified conceptual artwork, not as measured results.

Aesthetics alone are not a certification. Scientific figures should maximize evidence density without redundant charts; use a premium coordinated color palette, white background, clean typography, legible legends and axes, precise captions, accessible contrast, and balanced chart types. Bar charts alone are insufficient for many units.

## Completed figure improvements

**Scientific AI Figure 2**: Replaced the recently drafted four-panel version with a concise three-panel real-data plot: the observed 309-year annual sunspot record plus centered 11-year average, original lagged observation pairs, and actual 59-year held-out model predictions. Exact measured test RMSE values: 19.3824886840 for Ridge and 33.1750059073 for persistence. Training/validation/test windows: 178/60/59. The new figure avoids repeated time series and keeps the legend off the plotted measurements.

**Scientific AI Figure 3**: Replaced five-panel composition with four nonredundant panels using real Wisconsin Diagnostic Breast Cancer measurements. Real dataset: 569 original cases; training/validation/testing: 341/114/114. Held-out ROC AUC 0.9957010582. Classifier confusion matrix (benign, malignant) [[72,0],[2,40]]. Scientific improvements include actual feature names (not "F1" etc.), interpretable correlations, real held-out probabilities, and directly displayed classification errors in place of a near-saturated ROC curve.

Reproduction code committed:
- `chapters/scientific_ai/code/render_verified_figures_02_03.py`
- Original experiment scripts `figure_02_scientific_discovery.py` and `figure_03_scientific_ai_evolution.py`
- Executable workflow `.github/workflows/scientific-ai-real-data-figures.yml`
- Manuscript-ready text and matching revised captions `chapters/scientific_ai/code/scientific_ai_figures_02_03_manuscript_replacement.tex`

The enhanced graphics were rendered locally into PNG, PDF and SVG formats. Both PDF outputs were rendered to images and inspected. Complete source numerical evidence was matched to the plotted values. Local figure binaries are in the distribution ZIP; their commit to GitHub is not yet independently verified.

## Wider figure inventory and targeted inspection

A GitHub tree scan found **93 figure-related Python scripts** across 11 chapter families (including shared code). Detailed script-path inventory: `qa/BOOK_FIGURE_CODE_QUALITY_INVENTORY_20261109.json`.

A source inspection of Scientific AI Figures 2–12 is documented in `qa/SCIENTIFIC_AI_FIGURE_QUALITY_MATRIX_20261109.json`. Figure 11 is **HOLD** because its trained PINN heat-equation reference is an analytic/simulated field rather than an authentic observational dataset. Figures 4, 5, 7, 9, 10 and 12 are primarily conceptual/schematic artwork and must never masquerade as real measurements. Figure 6 needs chemical provenance validation and Figure 8 needs independent numerical/visual validation.

## Certification decision

**Two figure designs improved and underlying numbers verified. Global book publication QA remains HOLD.** The 93 script inventory does not include every manuscript figure, and no claim is made that every application or case study has now been checked, executed, or visually approved. Next requirements: per-section figure inventory, access to every original manuscript PDF/source asset, actual model-specific experimental results for missing units, independent numerical/visual/typographic inspection, GitHub code and output publication, caption/reference review, and final PDF compile audit.
