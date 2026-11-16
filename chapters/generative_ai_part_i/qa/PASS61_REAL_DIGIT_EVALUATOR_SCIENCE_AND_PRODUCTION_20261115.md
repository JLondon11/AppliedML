# Chapter 6 — Pass 61: independently executed original-digit evaluator control

**Date:** 2026-11-15  
**Repository:** `JLondon11/AppliedML`  
**Standard:** updated book-wide proportional R1–R10; empirical model results from original sources; retain genuine outputs and complete code; separate scientific certification from production approval.

## New actually executed original-data experiment

The original `code/section_6_20_1_latent_diffusion_digits/run.py` trains a `LogisticRegression(max_iter=2000,random_state=20261109)` classifier on a stratified source training split to evaluate whether generated images match the requested class. Pass 60 correctly identified conditional generated class agreement varying from 39% to 99%, but the original real-test **per-class** evaluator error was not juxtaposed with those model-output counts.

A new independent evaluation was executed on **actual `sklearn.datasets.load_digits()` real handwritten scans** (1,797). The exact original source `run.py` split was reproduced: seed 20261109, 1,257 train, 270 validation, 270 independent real test; no overlapping source indices. The held-out test index-array SHA-256 is `d945a001641ccbaecf31384feb1d1ba6e54c4d0dc33a3f6c9a40effe505514bb`. The original real-test evaluator independently scored **257/270 = 95.185185%**. The classifier was fitted only to authentic training scans; validation/test images were not used for training or model selection.

The saved prior generator output-count source `figure_6_44_class_counts_pass27.csv` had SHA-256 `1b3499a3fa7f93e0cd044ade2b827858393d6ad54c2bedf9cb99a9c7916b2fb0`, and each of ten original saved model-output classes had 100 outputs per conditional/unconditional arm. Existing totals reconciled: **778/1,000** conditional generated classifier matches versus **94/1,000** unconditional assigned-comparison-label matches. The generative model **was not retrained** in Pass 61: new training occurred **only for the original-real digit evaluator**.

| Digit class | Original independent real-test evaluator recall | Previously saved generated conditional agreement |
|---|---:|---:|
| 1 | 25/27 = 92.59% | 39/100 = 39% |
| 2 | 26/27 = 96.30% | 55/100 = 55% |
| 4 | 24/27 = 88.89% | 99/100 = 99% |
| 5 | 28/28 = 100% | 67/100 = 67% |
| 8 | 25/26 = 96.15% | 66/100 = 66% |

This **descriptive contrast** is scientifically useful: the severe generated digit-1/2/8 class-agreement deficits are unlikely to be explained simply by the evaluator being unable to recognize *original real examples of the same classes*. However, the real and generated populations, sample denominators and source dependence differ; there is **no paired statistical test** or claim of calibrated generator fidelity, and a high classifier match is not a visual-quality certificate. Some generated digits may be correlated, and a single generator seed cannot support cross-seed reliability estimates. Digit 4's 99% saved generated agreement exceeding 88.9% genuine real recall is an additional warning that the metrics are not directly comparable performance estimates.

## Committed evidence and publication candidate

Exact section path `chapters/generative_ai_part_i/code/section_6_20_1_latent_diffusion_digits/`:

- `audit_real_evaluator_control.py` — full runnable authentic-data classifier control, CSV/JSON results and figure generation.
- `original_evaluator_control/real_evaluator_vs_generated_class.csv` — all ten genuine held-out classifier class results and previously recorded generated class counts.
- `original_evaluator_control/real_evaluator_confusion.csv` — true original 270-image, 10×10 confusion matrix.
- `original_evaluator_control/metrics.json` — original train/validation/test split and measured results with explicit interpretation limits.
- `plot_real_evaluator_control.js` — deterministic vector-figure renderer from source-linked measured CSVs.
- `original_evaluator_control/real_classifier_control_summary.svg` — two-panel scientific original-data graphic (per-class agreement dot/dumbbell comparisons and original real-test confusion heatmap), not wide bars.
- `verify_original_evaluator_control.py` — numerical evidence and saved-generation class reconciliation.
- `README.md` — scientific interpretation, exact evidence and limitations.

The actual local run also produced an independently reviewed matplotlib figure (PNG/SVG), checkpoint-free actual output data, and execution log, included in the downloadable Pass 61 local evidence ZIP. GitHub CI was updated to re-execute the original-data classifier, compare metrics/rows against committed evidence and check vector-figure drift. **A completed new GitHub Actions run has not been independently verified.**

## Scientific and production determination

**CH6-D12 retains its existing Level A scoped scientific certificate** for an authentic-data conditioned latent-diffusion digits diagnostic; the new genuine-real evaluator control strengthens its R5/R6/R7 interpretation without upgrading the parent Stable Diffusion or natural-image application. No independent new scientific certificate is claimed, and no parent Chapter 6 application has been promoted.

The current authoritative scoped register records twelve separately scoped scientific certificates (eleven Level A, one Level B), all `PRODUCTION_HOLD`. The accessible GitHub Chapter 6 root still does **not** expose the latest LaTeX/PDF manuscript and final matching image assets. The Pass 37 note reports Figure 6.20 placed correctly within the industrial case study but is not the actual current visual/asset verification. The current full manuscript inventory baseline remains 48 figures, 17 tables, 11 listings and 14 parent studies; actual title-to-evidence mapping, final captions and visual integration need the current manuscript.

**Release status: HOLD**, pending verified Figure 6.20 source/caption/asset, genuine BDD100K multi-scene downstream perception for relevant claims, independently measured physical properties for claims about novel concrete, final manuscript production inspection, and title/claim-consistent R1–R10 for every full parent study. No fabricated numbers or data were used.
