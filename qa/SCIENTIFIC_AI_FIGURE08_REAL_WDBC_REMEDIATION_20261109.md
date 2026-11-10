# Scientific AI Figure 8 — Authentic WDBC scientific-figure verification

**Date:** 2026-11-09. **Scope:** Figure 8 real-data verification and figure-code improvement only; this does not certify the chapter or all case studies.

The UCI Wisconsin Diagnostic Breast Cancer dataset contains 569 genuine recorded cases with 30 measured features. The original observations were loaded through `sklearn.datasets.load_breast_cancer`; the exact float64 feature matrix digest was `6b202a2072f9a0385f405a8f8605b1b06f6f36ae6d23d9cd6cbbc0974a416bc7`. The stratified split retained 455 cases for five-fold CV training/development and 114 cases for an untouched held-out test. StandardScaler is fitted *inside* the training fold for each classical model; separate full-development-set fits produce held-out test predictions. Four prespecified models were compared, and their original-sample test predictions were saved.

| Model | Mean CV ROC AUC | CV sample SD | Held-out test accuracy | Held-out test ROC AUC |
|:---|---:|---:|---:|---:|
| Logistic regression | 0.995975 | 0.005622 | 0.982456 | 0.995701 |
| RBF SVC | 0.995562 | 0.005320 | 0.982456 | 0.995040 |
| Random forest (300 trees) | 0.990248 | 0.008200 | 0.947368 | 0.993717 |
| k-nearest neighbors (k=5) | 0.988235 | 0.009112 | 0.956140 | 0.978836 |

The revised figure comprises two complementary types of scientific evidence: five actual cross-validation measurements with mean/SD for each model, and four measured held-out ROC traces with quantitative AUC in the legend. The chart is produced from measured real-data prediction probabilities, never illustrative curves. The exact 20 fold records, 456 held-out predictions, four confusion matrices and numerical metric arrays are in the locally executed experiment ZIP.

**Local replication:** A complete second execution produced an exactly matching `results.json`; the accuracy and AUC arithmetic for saved predictions was independently checked and the PNG/PDF/SVG figure generated. The PDF was rendered and the PNG visually inspected.

**GitHub:** The revised generator is committed under `chapters/scientific_ai/code/figure_08_wdbc_baselines.py`; the numeric summary under `figure_08_wdbc_verified_summary.json`; the integrity check under `test_figure_08_real_data.py`; and the GitHub Actions workflow includes the new test. The full executed archive is a separate deliverable. The exact GitHub clean-checkout execution and final manuscript integration have not yet been verified.

**Separate outstanding problem:** Scientific AI Figure 11 currently uses an analytical heat-equation simulation with random PINN collocation points; it remains HOLD for a scientific experiment requiring genuine observed data.

## Figure 8 publication caption (proposed)
*Measured cross-validation performance and held-out classification for four models on the authentic Wisconsin Diagnostic Breast Cancer dataset. (a) Five stratified training-only ROC-AUC folds per architecture, with mean and sample standard deviation. (b) ROC curves calculated from each model's prediction probabilities for the same 114 held-out cases; 455 cases are reserved for development. Classifier performance is restricted to this source dataset and split.*
