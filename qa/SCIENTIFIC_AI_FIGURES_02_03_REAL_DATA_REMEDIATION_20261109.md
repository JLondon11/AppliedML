# Scientific AI — Figures 2 and 3 real-data remediation

Date: 2026-11-09. Scope: **two figures corrected**, not blanket Chapter 11 certification.

## Source-level violations verified

**Figure 2 (prior implementation).** The source created sinusoidal fields with `np.sin`, simulated a Lorenz trajectory, and generated `truth + np.random.normal(...)` points labeled sparse observations. No authentic dataset was loaded. These panels were unacceptable as empirical scientific results.

**Figure 3 (prior implementation).** The source assembled an analytic diffusion field and used `np.random.normal(...)` to produce fictitious noisy \"measurements\". It depicted purported autonomous experiment selections from kernel variance rather than genuine experimental measurements. This also failed the real-data requirement.

## Remediation executed

**Figure 2:** Replaced all panels with NGDC annual sunspot observations accessed through the statsmodels scientific dataset. There are **309 original annual observations (1700–2008)**. The 12-year one-step forecasting task has **178 training**, **60 validation**, and **59 held-out testing** windows. Training-only input standardization and validation RMSE choose Ridge alpha = 10 from a fixed grid. The executed held-out RMSE is **19.3824886840**, against **33.1750059073** for last-observation persistence; held-out MAE is **14.0901769265**. Four panels now show original observations, observed 11-year rolling averages, observed lag relationships, and actual fitted model predictions. All panels are derived from measured data, not invented values.

Source: `chapters/scientific_ai/code/figure_02_scientific_discovery.py`.

**Figure 3:** Replaced the simulated diffusion and fictional experiment panels with genuine digitized cell-nucleus measurements from the UCI Wisconsin Diagnostic Breast Cancer dataset, accessible through scikit-learn: **569 records and 30 recorded features**. Stratified disjoint splits contain **341 training**, **114 validation**, and **114 held-out testing** records. Training-only standardization; logistic regression penalty parameter C selected with validation malignancy F1. Selected C = 1; held-out accuracy **0.98245614035**, malignancy F1 **0.97560975610**, and ROC AUC **0.99570105820**. Confusion matrix (true benign / malignant) is **[[72,0],[2,40]]**. The five panels now show observed biomedical measurements, training-set empirical feature correlations, observed validation performance, genuinely predicted held-out probabilities, and a measured ROC curve.

Source: `chapters/scientific_ai/code/figure_03_scientific_ai_evolution.py`.

## Execution checks

- Both replacement scripts executed locally against the authentic datasets.
- Each experiment was independently rerun; the complete saved numerical JSON result matched its first run **exactly**.
- Every plotted result can be regenerated via its committed source code; PNG, PDF and SVG files were generated and visually inspected in the local package.
- Figure-specific test scripts and committed numerical summaries are located adjacent to the generators in GitHub.
- The intended manuscript filenames are preserved: `figure_02_evolution_scientific_discovery` and `figure_03_evolution_scientific_ai`. Default output directory is `chapters/scientific_ai/figures`.

**Publication limitation:** The generated binary figure assets and full prediction arrays are provided in the archived execution package, but their presence in the GitHub repository has **not** been verified. The complete Chapter 11 LaTeX manuscript was **not** rebuilt or checked to ensure all figure references, titles, discussion and citations align with these new scientific meanings. Both units remain **MANUSCRIPT-INTEGRATION HOLD**.

## Additional identified risk — Figure 11

`chapters/scientific_ai/code/figure_11_pinn_residual.py` trains a PINN against an analytically defined heat equation using randomly sampled collocation points, rather than against an authentic measured scientific dataset. It is legitimate mathematics/code execution, but under the book's requested **real-data empirical-result** standard it is **NOT VERIFIED AS COMPLIANT**. Replace its claimed experimental evidence with an observation-grounded PINN benchmark or clearly separate a mathematical derivation from empirical results. Do not label synthetic PDE targets as measurements.

## Wider audit status

The repository-wide audit of other applications/case studies and figure artifacts remains open. Every empirical figure must ultimately be matched to its actual dataset, run code, measured outputs, output asset, figure caption, and manuscript use. A figure generator existing on GitHub is not scientific certification.
