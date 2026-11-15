# Chapter 2 Pass 2 — source-verified real-data scientific remediation (2026-11-14)

## Authoritative manuscript and registration correction

The uploaded `Chapter_2_Deep_Learning_Part_I_Pass2_Revised_Package.zip` contains a distinct **117-page Chapter 2** source (`Chapter_2_Deep_Learning_Part_I_REVISED_QA_PASS2.tex`), with **five named empirical application/case-study headings**: nonlinear two moons; deterministic convolution on real Digits; real Digits MLP classification; real Diabetes MLP regression; and repeatability across random seeds on generated two moons. The older **eleven-unit** register concerns a different edition of Deep Learning Part I; its LSTM/GRU sequence classification and UCI HAR titles **are absent** from the uploaded current manuscript. Preserve its historical findings, but do not claim it is the live five-unit certification matrix.

The actual current-book five-unit proportional R1–R10 register is at `CH2_PASS2_CURRENT_MANUSCRIPT_R1_R10_20261114.json`. Scientific and production certifications are separately recorded.

## Genuine-data experimental execution and correction of published inconsistencies

The source scripts included in the user-supplied Pass 2 archive use internally early-stopped MLPs. When these exact original scripts were executed, the actual outputs were **0.95556** Digits accuracy (versus **0.9689** printed) and **59.16976** diabetes RMSE (versus **58.87** printed). The printed experimental numbers were not reliable as written.

For a cleaner, explicitly disclosed fixed-budget numerical study, we disabled `early_stopping` (to avoid mixing a scaler fitted on an entire outer training partition with an internal validation subsplit), retained the original architecture, original-data stratified/test split, optimizer, seeds and settings, and reran the data and comparison models. This **changes the scientific protocol**, and the actual manuscript code and captions were updated accordingly. We did **not** misrepresent it as replication of the originally printed scores.

### Digit classification: real original 8×8 handwritten images

- Actual scikit-learn Digits observations **1,797**, stratified **1,347 train / 450 independent outer test**, seed 1729.
- Training-only standardization; executed 64–32 MLP (maximum 250 iterations, stopped at solver convergence after **110**) and matched standardized logistic regression.
- **Actual MLP:** accuracy **0.9800**, macro-F1 **0.9800376718**, macro OvR AUC **0.9988058356**.
- **Actual logistic reference:** accuracy **0.9755555556**. This is only two fewer correct images on the single test split, not evidence of general architectural superiority.
- Saved original source record indices, complete 450 target labels, MLP probabilities/predictions and baseline predictions, 110 actual training-loss values.
- Regenerated **Figure 2.31** training-loss/accuracy, **Figure 2.32** observed confusion matrix and class recall, **Figure 2.33** measured reliability and actual misclassified original images (**9 errors**).

### Diabetes regression: real observed patient progression data

- Original scikit-learn Diabetes dataset **442**, **331 train / 111 independent test**, seed 1729.
- Training-only standardization; fixed-budget 64–32 MLP regressor and matched Ridge regularized regression.
- **MLP RMSE 58.2164949**, MAE **49.2404926**, R² **0.4421587**; **Ridge RMSE 57.1243682**, R² **0.4628923**.
- MLP hit the **700-iteration cap without optimizer convergence**; reported and kept as a scientific limitation.
- **Figure 2.34** plots actual observed-versus-predicted points, actual regression residuals, and both methods' measured error distributions across 111 test records.
- The MLP **does not outperform Ridge** in this observation-grounded example.

The locally executed full experiment was **rerun twice**: `measured_results.json` SHA-256 in both runs was `b57606bfa7071fcb5724c5f854707419b55feabab52ca04b8782401078f50846`. Independently recomputing original-data SHA-256, disjoint outer splits, all held-out classification scores, regression errors, and source figure files passed.

## Actual manuscript integration and PDF QA

- Corrected numerical tables, model training descriptions, conclusion scope, exact source-code scripts, and output figures in the **actual supplied Pass 2 LaTeX**.
- Figures **2.31–2.34** placed under their original chapter captions and labels, all generated directly from real execution arrays; PDF/PNG/SVG versions retained.
- Enlarged these four experimental figures to 0.96 text width and examined rendered PDF pages **88, 89, 92**.
- Two successful full `pdflatex` passes, **117 pages**, final PDF SHA-256 **`d0916a64c46ad02b944c102ad78aae68226fcce29a19aae8c76f6d22f82d002b`**. No undefined citation/reference warnings.
- Local full deliverable: `Chapter2_Pass3_RealData_Figures_MasterQA_20261114.zip`. The complete revised LaTeX, compiled PDF and figure binaries are distributed in the ZIP, but **have not been verified pushed to GitHub**.

## Scoped proportional R1–R10 decisions

**CH2-P2-03 (Digits MLP)** and **CH2-P2-04 (Diabetes MLP): LEVEL B scientific PASS for the exact offline observed-data models, fixed train/test partitions, actual saved predictions and published diagnostics.** R1–R10 are applicable and satisfied for these limited empirical scopes. The Diabetes nonconvergence and inferior score versus Ridge are explicitly reported, rather than suppressed.

**CH2-P2-01 Two Moons** and **CH2-P2-05 Reproducibility across Random Seeds** remain scientific **HOLD** because they use generated two-moons rather than authentic observational data as their actual title's experimental dataset. They can serve as explicitly labeled simulated pedagogical examples, but do not acquire empirical certification merely by changing assessment level.

**CH2-P2-02 convolution on a real digit** remains a scoped Level A **candidate/HOLD**, pending independent reconciliation of the fixed-kernel response matrices, exact image pixel identity and Figure 2.19's displayed values with executed code.

**Whole chapter Master QA remains HOLD** (2 scoped B-level passes; 3 named units unresolved). The rest of the 35 figures and 23 code listings still require their own source- and publication-level checks.

## Next scientific work

Priority 1: verify real Digits convolution image source index and all fixed-kernel numerical outputs, likely yielding a proportionally appropriate Level A certificate. Priority 2: replace simulated two-moons repeatability with existing authentic Digits multiseed observations and actual measured variation, only with properly revised text and figures. Priority 3: if human-activity classification is restored as a named application, use the **original UCI HAR** sensor windows and participant-disjoint evaluation rather than unrelated forecasting proxies. The official UCI dataset could not be transferred into this restricted local runtime in the current pass; **no unexecuted HAR model is certified**.

**Source code publication:** `chapters/deep_learning/part_i/experiments/pass2_real_digits_diabetes/run.py` and `test_outputs.py` are complete runnable experiment and numeric verification sources in AppliedML. A GitHub-hosted clean-checkout run has not yet been independently confirmed.
