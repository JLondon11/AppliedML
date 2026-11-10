# Chapter 11 manuscript real-data figure integration — 2026-11-10

## Source and scope

The actual archived Chapter 11 LaTeX source `all_chapters_delivery_v2/11_Scientific_AI/Scientific_AI_InProgress.tex` was extracted from the supplied `All_Chapters_Complete_12.zip`, and its original Scientific AI figure directory was recovered. This pass integrated existing **authentically measured** experiments into the real book chapter, rather than only creating isolated figure scripts.

The deliverable `Scientific_AI_Chapter11_RealData_Integrated_20261110.zip` contains the complete revised LaTeX, the complete rebuilt chapter PDF, original figure assets, the two replacements in PNG and PDF, and underlying experiment records. The package is provided separately through the book-editing session; **the full edited LaTeX and binary figure assets are not claimed to be committed to GitHub** because repository push access for local binary assets is unavailable in the current runtime. The executable figure scripts themselves are committed to AppliedML.

## Figure 11.8 — actual WDBC classification results

- Replaced the originally referenced `Figures/Figure_11_08_WDBC_Classical_Classifier_Benchmark.png` with the observed five-fold ROC-AUC and held-out ROC figure generated from genuine Wisconsin Diagnostic Breast Cancer measurements (569 records, 30 observed features).
- The real development/test partition is 455/114. Actual test predictions are saved for four prespecified classifiers, and the chapter prose was updated to include their measured results.
- Corrected the caption to match **two** actually rendered panels, not the prior three-panel workflow description. Retained the original LaTeX figure label `fig:scientific_ai_07_application_classical_machine_learning_baseline_comparison_w`.
- Corrected Listing 11.3: the earlier excerpt standardized the entire development set before cross-validation (information leakage) and specified classifier models inconsistent with the measured figure. The replacement short listing fits `StandardScaler` inside each training fold using a scikit-learn pipeline; the full measured four-model implementation remains available on GitHub. Removed an unbreakable minipage wrapper that caused a large blank region.
- The relevant figure was placed after background and quantitative discussion rather than floating ahead of its application section.

## Figure 11.11 — real experimental heat measurements

- Replaced `Figures/Figure_11_11_PINN_Residual_versus_Field_Error.png`, previously an analytical simulation, with the actual eight-sensor heated aluminum-rod dataset from `TariqSawaqed1/Heat-Conduction-Aerospace-Lab`, 168 real timestamps and **1,344 measured temperatures**.
- Whole-sensor split: 1,008 training measurements across six sensors; 168 validation measurements at CH3; 168 held-out test measurements at CH6.
- Original trained model test RMSE was 0.1103°C (data-only), 0.1856°C (regularization strength 0.002), and 0.4937°C (0.02). The physics-regularized models performed **worse**; the negative result is retained. No invented thermal field or measured material-diffusivity claim is made.
- Replaced the inaccurate residual-field caption with actual four-panel figure descriptions. Added three background/method/results paragraphs, complete-sensor split, test limitations, and the original laboratory source bibliography. Retained existing label `fig:scientific_ai_05_pinn_residual_analysis`, so figure numbering remains Figure 11.11.
- The figure is now placed after the experiment explanation, before discussion of wider PINN solver-evaluation protocols.

## Build and visual QA

- The full actual chapter compiled successfully twice with pdflatex after final modifications.
- Final PDF is **157 pages**, SHA-256 `266fe6f4a967e99d7b5aed33ca69d105574daa5b9b2f1d215e34b7ae042d7408`.
- Second complete run produced **no undefined citations or cross-references**.
- Checked 55 explicit graphics references: all literal file paths are present, one extra token `Figures/#1` is a macro argument rather than a missing file.
- No duplicate LaTeX labels.
- Visually inspected corrected Figures 11.8 (PDF page 32) and 11.11 (page 40) and neighboring page content for correct graphic/caption/topic alignment; fixed float ordering and WDBC code-listing whitespace issue.

## Remaining global certification holds

1. Figure 11.3 in the archived chapter **still** describes noisy experimental observations that were artificially generated in its old mathematical figure construction. It must be remediated before certifying all Scientific AI figures.
2. Many additional applications, figures and code listings in Chapter 11 require independent source-to-data, caption, scientific and visual checks.
3. This archive may not be the newest evolving canonical chapter from other separate book-editing conversations; any later merge must reconcile sources carefully.
4. The revised full chapter LaTeX and figure assets exist in the delivered ZIP but have **not** been verified as committed to GitHub. Code sources and their tests are already committed.

**Decision:** Figure 11.8 and Figure 11.11 source-to-manuscript integration complete in the packaged draft. **Whole-chapter Master QA: HOLD.**
