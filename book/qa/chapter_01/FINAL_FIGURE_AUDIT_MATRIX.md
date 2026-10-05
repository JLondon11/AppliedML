# Chapter 1 Final Figure Audit Matrix

Scope: all 21 currently retained Chapter 1 figure stems on `book-publication-recovery-final`.

Acceptance standard: a figure is FINAL ACCEPT only after it passes all of:
1. chapter relevance;
2. scientific correctness;
3. provenance/reproducibility;
4. no data leakage / protocol error;
5. no semantic duplication;
6. correct caption/panel mapping;
7. quantitative legend/colorbar where needed;
8. compact publication aesthetics;
9. correct size/whitespace in the compiled chapter;
10. explicit discussion in surrounding prose.

| Figure stem | Current decision | Scientific audit |
|---|---|---|
| ch01_credit_card_fraud_imbalance | ACCEPT-SCIENCE; final visual/layout pass pending | Real OpenML/ULB credit-card data, chronological split, reproduced PR/ROC/threshold metrics. |
| ch01_medical_pneumoniamnist | ACCEPT-SCIENCE; final visual/layout pass pending | Real MedMNIST v2 PneumoniaMNIST benchmark with explicit nonclinical-use limitation and provenance. |
| ch01_optimizer_conditioning_sensitivity | ACCEPT-SCIENCE; final visual/layout pass pending | Controlled numerical experiment on real WDBC data; conditioning manipulation and failure regime explicitly defined. |
| ch01_optimizer_selection_benchmark | ACCEPT-SCIENCE; final visual/layout pass pending | Same regularized logistic objective across optimizers, repeated fixed splits, held-out metrics and uncertainty. |
| ch01_predictive_maintenance_ai4i | REPLACE / RECONCILE | Current repository naming is inconsistent with the stronger CWRU bearing-data provenance/generator. Final Foundations figure should use the real CWRU vibration evidence and a matching stem/caption/provenance package, not retain an ambiguous AI4I-labelled asset. |
| ch01_smart_grid_load_forecasting | ACCEPT-SCIENCE; final visual/layout pass pending | Real UCI household electric-power data with chronological lag-based forecasting protocol and provenance. |
| ch01_turbofan_prognostics | REPLACE | Current artwork is an explicitly labeled numerical reconstruction. For a final Application figure, the best defensible scientific choice is a protocol-matched real C-MAPSS rendering/experiment if the source data are available. |
| figure_01_03_model_capacity | ACCEPT-SCIENCE; final visual/layout pass pending | Controlled polynomial-capacity experiment on the same sampled regression problem; no empirical benchmark claim. |
| figure_01_04_inductive_bias | ACCEPT-SCIENCE; final visual/layout pass pending | Multiple interpolants fit the same finite observations and diverge between samples; valid computational illustration of inductive bias. |
| figure_01_06_learning_curves | REGENERATE | Audit found train/validation leakage because scaling was originally fit before the split. Generator fixed in commit `7617159fc8ac7cb253ad034e460bf25d5bc6649f`; figure must be regenerated before acceptance. |
| figure_01_07_bias_variance | ACCEPT-SCIENCE; final visual/layout pass pending | Monte Carlo bias/variance decomposition from repeated noisy polynomial-regression samples. |
| figure_01_08_representation_preprocessing | ACCEPT-SCIENCE; final visual/layout pass pending | Train-only scaler/PCA fit, quantitative conditioning and held-out ROC-AUC diagnostics. |
| figure_01_10_dimensionality | ACCEPT-SCIENCE; final visual/layout pass pending | Real sklearn digits data with PCA, t-SNE, and spectral embedding; no invented performance claim. |
| figure_01_11_hpo | REGENERATE | Audit found preprocessing leakage because standardization was originally performed once before cross-validation. Generator fixed to use fold-internal pipelines in commit `7617159fc8ac7cb253ad034e460bf25d5bc6649f`; figure must be regenerated. |
| figure_01_12_metrics | ACCEPT-SCIENCE; final visual/layout pass pending | Real WDBC held-out classifier evaluation: ROC, PR, calibration, and confusion-matrix diagnostics. |
| figure_01_13_cross_validation | REVISE | Split-assignment computation is valid, but fold IDs are encoded categorically without a clear categorical legend/direct labels. Redesign for clearer publication interpretation. |
| figure_01_15_governance_adult_audit | ACCEPT-SCIENCE; final visual/layout pass pending | Real UCI Adult test data, sensitive audit attributes excluded from training features, descriptive group-conditional error/calibration diagnostics, no unsupported normative fairness threshold. |
| figure_01_17_clustering | ACCEPT-SCIENCE; final visual/layout pass pending | Actual k-means, hierarchical, DBSCAN, and GMM outputs on a matched controlled clustering dataset. |
| figure_01_18_svm | ACCEPT-SCIENCE; final visual/layout pass pending | Actual linear/RBF SVM decision surfaces and support vectors on a controlled nonlinear benchmark. |
| figure_01_19_tree | ACCEPT-SCIENCE; final visual/layout pass pending | Actual trained decision tree, decision surface, feature importance, and split-count diagnostics. |
| figure_01_20_ensembles | REGENERATE-MINOR | Scientific design is valid; SVC probability calibration path lacked explicit deterministic seed. Generator fixed in commit `7617159fc8ac7cb253ad034e460bf25d5bc6649f`; regenerate before final acceptance. |

## Current conclusion

No blanket FINAL ACCEPT is authorized yet.

Current categories:
- 15 figures are scientifically acceptable in design/provenance and await final visual/layout/caption inspection.
- 3 figures require regeneration after protocol/reproducibility fixes (1.6, 1.11, 1.20).
- 1 figure requires visual-semantic redesign for categorical clarity (1.13).
- 2 Application figures require replacement/reconciliation (predictive-maintenance naming/evidence package and turbofan/C-MAPSS).

The final publication PDF must use only figures that have been promoted to FINAL ACCEPT after regeneration/replacement and compiled-page visual inspection.
