# Chapter 1 Final Figure Audit Matrix — Asset-Level Freeze

Repository branch: `book-publication-recovery-final`

This matrix freezes the **22 current retained Chapter 1 figure families** after individual scientific, provenance, redundancy, and rendered-artwork inspection. Every retained family has matching PNG, SVG, and PDF files.

**Important:** `FINAL ACCEPT — ASSET` means the standalone scientific artwork is accepted. A figure is not publication-final until the integrated Chapter 1 PDF also passes caption fidelity, sequential numbering, section placement, size/whitespace, accessibility alt text, and substantive in-text discussion.

| Retained figure stem | PNG blob SHA | Asset decision | Scientific / visual basis |
|---|---|---|---|
| `figure_01_03_model_capacity` | `f4e6e181925f04d5051469425fde6742db91cd97` | FINAL ACCEPT — ASSET | Controlled same-sample polynomial capacity experiment clearly distinguishes underfit, intermediate fit, and unstable high-capacity behavior. Caption must explicitly explain edge instability of the highest-degree fit. |
| `figure_01_04_inductive_bias` | `c530ed6423d822ba7cfa7917a7c9a1373f05eb82` | FINAL ACCEPT — ASSET | Multiple hypotheses fit the same finite observations but diverge between/outside samples; direct computational demonstration of inductive bias. |
| `figure_01_06_learning_curves` | `239da4ac06f82778405b8efac0585375c002962e` | FINAL ACCEPT — ASSET | Redesigned after leakage and visual audit. Five-fold learning curves now show distinct underfit, well-fit, and high-variance regimes with training/CV uncertainty. |
| `figure_01_07_bias_variance` | `cd4c2d0352f02069a4217b9b4b1440ff86ea8148` | FINAL ACCEPT — ASSET | Monte Carlo polynomial experiment now explicitly separates squared bias, variance, irreducible noise, and expected test MSE. |
| `figure_01_08_representation_preprocessing` | `a3b6c57e64e64d2a13396c6d0e716d5495c43741` | FINAL ACCEPT — ASSET | Train-only scaling/PCA; quantitative feature-scale, condition-number, and held-out ROC-AUC diagnostics. |
| `figure_01_10_dimensionality` | `d0d18185d6075d07fb2b2d311a8362495dc80053` | FINAL ACCEPT — ASSET | Real sklearn digits; PCA, t-SNE, and Isomap. Isomap replaced a collapsed spectral-embedding panel after visual audit. |
| `figure_01_11_hpo` | `a6380f44065d5fbc9327f6e74c643578cd756b67` | FINAL ACCEPT — ASSET | Fold-internal preprocessing pipeline eliminates prior CV leakage; matched-budget random/grid/coarse-to-fine search trajectories are clear and quantitative. |
| `figure_01_12_metrics` | `265c979499927e6a832a93baaf3871a4f8ec1983` | FINAL ACCEPT — ASSET | Real WDBC held-out ROC, PR, calibration, and confusion-matrix diagnostics; confusion matrix now has categorical class ticks and quantitative count colorbar. |
| `figure_01_13_cross_validation` | `a5ac883efaa3dca2e145a22eeb3300ae4b68ed25` | FINAL ACCEPT — ASSET | Redesigned from dense fold-ID heatmap to readable train/validation/outer-test assignment matrices with direct categorical legend. |
| `figure_01_15_governance_adult_audit` | `cb4ebe67ac538b127dae3d38fc60c77d41c6c484` | FINAL ACCEPT — ASSET | Real UCI Adult test data; audit attributes excluded from training; descriptive group-conditional error/calibration diagnostics without unsupported normative fairness threshold. |
| `figure_01_17_clustering` | `129791386af2af2f728d000ba36dd25be27f6ced` | FINAL ACCEPT — ASSET | Actual k-means, agglomerative, DBSCAN, and GMM outputs on the same controlled dataset. Caption must state cluster colors are algorithm-local labels, not cross-panel identities. |
| `figure_01_18_svm` | `d4ddcb180453c81158880ea7fc2dbbe7348256df` | FINAL ACCEPT — ASSET | Actual linear/RBF SVM decision functions and support vectors on a matched nonlinear benchmark. |
| `figure_01_19_tree` | `b4e9b3d4b659a57c5116e2e9ad0e6c31d0a3f897` | FINAL ACCEPT — ASSET | Decision surface plus capacity/generalization diagnostic; weak split-count panel replaced by training/validation accuracy versus depth and leaf count. |
| `figure_01_20_ensembles` | `773fe9def03e03af98f10b5147263afdc1e0bdb4` | FINAL ACCEPT — ASSET | Actual bagging, boosting, random-forest, and stacking decision surfaces; deterministic probability model configuration. |
| `figure_01_24_predictive_maintenance_cwru` | `352386a0841a4bc71e4dcfaadd87809eb7869b46` | FINAL ACCEPT — ASSET | Real CWRU vibration data. Correct file-variable mapping, 48-kHz normal-to-12-kHz fault sampling alignment, file-level 0/1/2-hp train versus unseen 3-hp test. Perfect scores are explicitly framed as restricted-benchmark limitation. |
| `figure_01_25_medical_preprocessing` | `257c9835c2ee3821f60afeb25e7c615cfeebeb9a` | FINAL ACCEPT — ASSET | Real scikit-image immunohistochemistry sample, grayscale conversion, and computed Otsu binary mask; valid preprocessing demonstration, not clinical evidence. |
| `ch01_optimizer_selection_benchmark` | `b6552175e0779bee874d69db1d55a23b383dd079` | FINAL ACCEPT — ASSET | Repeated-split real WDBC optimizer benchmark; same objective/model across methods; convergence plus held-out uncertainty. |
| `ch01_optimizer_conditioning_sensitivity` | `32dfbd7251114d669910e740cb5391f59adc57fa` | FINAL ACCEPT — ASSET | Controlled conditioning perturbation on real WDBC data; clearly exposes regime-dependent L-BFGS behavior and threshold cap. |
| `ch01_credit_card_fraud_imbalance` | `68ea787aead8975eeee998963c3b5e3d2f6411c9` | FINAL ACCEPT — ASSET | Real OpenML/ULB credit-card data with chronological split; precision-recall and fixed-review-budget diagnostics appropriate for extreme imbalance. |
| `ch01_medical_pneumoniamnist` | `1f60c6de1935db60d238d6332dd4bb484f4881f5` | FINAL ACCEPT — ASSET | Real MedMNIST v2 PneumoniaMNIST examples and reproduced model precision-recall comparison; explicitly nonclinical educational/research scope. |
| `ch01_smart_grid_load_forecasting` | `fd736cac4b84e50f5c59a3c1f013eb52cb4b5668` | FINAL ACCEPT — ASSET | Real UCI household-power data; causal lag features, chronological holdout, strong seasonal baseline, readable held-out-week date axis and hour-of-day error diagnostic. |
| `ch01_turbofan_cmapss_fd001` | `7017c9dde78ab2210fd12c87723c8bf69bbc2abc` | FINAL ACCEPT — ASSET | Real NASA C-MAPSS FD001. Actual sensor trajectories plus reproduced 100-engine true-vs-predicted RUL diagnostic; RMSE 19.05 cycles, MAE 14.39 cycles, restrained palette. |

## Removed / superseded figure families

The following are intentionally excluded from the final Chapter 1 figure inventory:
- duplicate learning-paradigm and feature-workflow figures;
- deployment/CI-CD, deep-network, backpropagation, autonomous-driving, climate, edge-AI, and precision-agriculture figures moved to their conceptually correct later chapters;
- synthetic AI4I predictive-maintenance figure, superseded by the real CWRU experiment;
- numerical/synthetic turbofan rendering, superseded by the real NASA C-MAPSS FD001 experiment;
- duplicate UCI smart-grid figure package, superseded by `ch01_smart_grid_load_forecasting`.

## Remaining figure gate

All 22 standalone figure assets are now accepted. The Chapter 1 **publication** figure gate remains open until the integrated manuscript confirms:
- every figure appears exactly once;
- numbering is sequential and captions match the final artwork;
- every panel is described in the unified caption;
- each figure is explicitly discussed in prose;
- figure sizing and whitespace are consistent on the compiled page;
- no figure/table/listing collisions or large vertical gaps occur;
- alt-text inventory covers all 22 retained figures.
