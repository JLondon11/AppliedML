# Chapter 2 Deep Learning Part I — Pass 4 authentic-data scientific certification

Date: 2026-11-14. Governing policy: proportional `SCIENTIFIC_EXPERIMENT_CERTIFICATION_STANDARD.md`, R1–R10, Level A/B/C corresponding to supported title scope.

## Source integrity and important distinction

Authoritative user-uploaded Pass 2 source: `Chapter_2_Deep_Learning_Part_I_Pass2_Revised_Package.zip`, updated through Pass 3 to the local Pass 4 package `Chapter2_Pass4_All5_RealData_20261114.zip`. It has **five named application/case-study units**; older eleven-unit experimental audits correspond to a different draft and must not be reported as the active count.

**Scientific result:** Five named units now have authentic-data observations, executed algorithms and actual output figures within the specifically rewritten, accurately limited experimental scopes. These are **two Level A and three Level B scoped scientific certificates**, not five Level C benchmark claims. No absent high-end research trial is required for a focused technical demonstration.

| Unit | Assigned level | Exact certified experimental scope | Result |
|---|---|---|---|
| CH2-P2-01 | A | MLP with training-only two-component PCA for original handwritten digit 3 versus digit 8, 249 training / 108 held-out images, measured output vs matched logistic reference | MLP accuracy 0.888889, logistic 0.870370, ROC AUC 0.972213, Figure 2.5 |
| CH2-P2-02 | A | Three exact 3×3 derivative/averaging spatial correlations on *original* 8×8 Digits image 0 with reflect boundary, actual source pixels and full response matrices | Mean absolute responses 1.42578125 / 3.23828125 / 4.59375, Figure 2.19 |
| CH2-P2-03 | B | Original 1,797 handwritten Digits images, 1,347 training / 450 held-out, 64–32 MLP vs matched logistic comparison | Accuracy 0.980000 vs logistic 0.975556, Figures 2.31–2.33, previously executed and published |
| CH2-P2-04 | B | Original 442 diabetes progression observations, 331 train / 111 held-out; actual 64–32 MLP vs Ridge | RMSE 58.2165 MLP vs 57.1244 Ridge, Figure 2.34, nonconvergence/negative model-comparison retained |
| CH2-P2-05 | B | Twelve genuine 24–24 MLP runs at seeds 0…11 on identical original handwritten Digits images, 1,347 train / 450 held-out | Accuracy mean 0.969074, sample SD 0.004589, range 0.957778–0.975556; 5,400 genuine held-out predictions, Figure 2.35 |

The original **two-moons** synthetic-data empirical application was **not certified** merely by reinterpreting criteria. Instead, its title, dataset, executed model, figure, caption, explanatory text, evaluation and conclusions were **replaced** by the original-image 3-versus-8 experiment. The old two-moons code was excluded from the new local delivery. Historical figure file basenames remain for LaTeX compatibility; they are not scientific titles or claimed source datasets.

### Numerical verification

The real original-Digits matrix SHA-256, all 3 actual 8×8 spatial maps and 3×3 kernels, fixed 1,347/450 stratified index partitions, 12 sets of 450 genuine predicted digit labels, stored per-epoch losses, seedwise accuracy, macro-F1 and sample SD have been independently recomputed. The 3-versus-8 original records and disjoint 249/108 partition, all 108 held-out predicted labels/probabilities, logistic reference and actual ROC AUC were also reconciled.

Two successive local executions of **each** of the two new experiment scripts produced **byte-identical numeric JSON**. The local run includes source scripts `code/verify_convolution_and_seed_digits.py`, `code/real_digits_nonlinear_2d.py`, and numeric integrity tests.

### Manuscript and figure production

- All three newly modified figures use real observed image information and/or genuine model predictions: Figure **2.5**, **2.19**, and **2.35**. Existing real-data Figures **2.31–2.34** remain from Pass 3. The chapter also contains pedagogical architecture and mathematical figures, which need not masquerade as empirical data.
- Figure 2.5 original digit photographs/learned probability surface/held-out outcomes, source text and caption replaced. Figure 2.19 exact pixel maps and kernels enlarged for printed legibility, matching source, labels and responses. Figure 2.35 actual full 12-seed accuracies, optimization traces and macro-F1 plots. The old synthetic outcome table and sample code listing were replaced with real-data values/implementation.
- Complete `pdflatex` builds twice, final PDF **118 pages**, Figures 2.5 / 2.19 / 2.35 on **PDF pages 18 / 52 / 98**.
- PDF SHA-256: `57b0d91fd0bef95de969edd9c8fa362378c1e3a5d2b7e030b8e087fe63f1cc89`. Original full numeric Digits 12-seed evidence JSON SHA-256: `0b0d1f223e90bf2355ed8a2fd944cdd3cbda2e773c0047b0c74b48da25efa888`.
- Final LaTeX log has no undefined citations/references. An existing duplicate hyperref frontmatter page-anchor warning remains; this is a separate production cleanup item, not a fabricated science metric.
- Final revised LaTeX, compiled PDF, original image-derived PNG/PDF/SVG assets, full prediction JSON, and evidence source are in the local ZIP. These complete binary and document assets are **not yet verified committed to GitHub**.

### GitHub code

- `chapters/deep_learning/part_i/experiments/pass4_real_digits_nonlinear/run.py` and `test_outputs.py`.
- `chapters/deep_learning/part_i/experiments/pass4_real_digits_convolution_seeds/run.py` and `test_outputs.py`.
- `.github/workflows/chapter2_pass4_real_data_experiments.yml` runs both authentic-data studies and archives all figures and raw predictions, **but a successful CI run has not been confirmed**.
- `chapters/deep_learning/part_i/qa/CH2_PASS2_CURRENT_MANUSCRIPT_R1_R10_20261114.json` contains current five-unit scoped PASS statuses.

**Decision: Five of five specifically named empirical units in the current revised manuscript SCIENTIFICALLY CERTIFIED within their precise A/B scopes.** This does **not** certify all other unreviewed pedagogical paragraphs or all chapter figures, nor does it constitute publication certification. **Chapter 2 final production gate: HOLD. Book-wide Master QA: HOLD.** Next task is entire Chapter 2 figure, listing, reference, and layout production audit, then canonical manuscript/binary GitHub publication.
