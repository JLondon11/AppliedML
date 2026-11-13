# Chapter 6 — Pass 49: proportional scope-specific scientific certification

**Date:** 2026-11-13. **Governing standard:** repository-root `SCIENTIFIC_EXPERIMENT_CERTIFICATION_STANDARD.md` and Chapter 6 `SCIENTIFIC_CERTIFICATION_POLICY.md` (scientific certification separated from manuscript production).

## Result and correct counting

Three **genuine original-data executed Level A demonstrations** have been adjudicated **SCIENTIFIC_CERTIFIED (all R1–R10 PASS within their explicitly narrow methods and claims)**:

1. **CH6-D01** — retrospective Brownlee industrial-process static Gaussian-process emulator versus ridge and training-mean controls.
2. **CH6-D02** — 2D-latent handwritten-digit VAE versus PCA held-out reconstruction comparison.
3. **CH6-D03** — real Mauna Loa CO₂ conditional-VAE prior-predictive eight-week forecasting with honest miscalibration error.

**This does NOT mean 3 of the 14 full named case studies/applications are certified.** Broad digital-twin, climate-system, synthetic-driving or full generative research claims still require their own domain-specific evidence. The previous 14-unit Master QA HOLD is **not** overridden by the embedded demonstration certificates. **Production remains HOLD** pending actual latest chapter PDF/TeX and figure integration checks.

Authoritative distinct scope/status source: `qa/CH6_SCOPED_SCIENTIFIC_VS_PRODUCTION_REGISTER_20261113.csv`. Historical `CERTIFICATION_REGISTER.csv` and Pass 46–48 reports are preserved, not retroactively rewritten.

## CH6-D01 — genuine Brownlee static process emulation

**What was executed anew:** independent implementation of the **same native Gaussian-process kernel, sklearn fitting protocol, fixed random state and training-only scaling for all 21 leave-one-out held-out predictions** on the original statsmodels Brownlee stack-loss observations. This was a new local measured experiment replay, not software fixtures. The true dataset source has 21 original measured records of AIRFLOW, WATERTEMP, ACIDCONC, STACKLOSS; SHA256 for little-endian float64 `[STACKLOSS,AIRFLOW,WATERTEMP,ACIDCONC]` is `d999d98d68f6cd1ad64dec41476b32961a63e60a3e81433bca361601b5612e6d`.

| Original real-data 21-fold leave-one-out estimator | RMSE (stack loss units) | MAE |
|---|---:|---:|
| Gaussian-process Matérn 3/2 plus fitted white noise | 4.118931861605197 | 2.966674493896961 |
| Ridge regression | 3.647664958998736 | 2.934435485017122 |
| Training-mean reference | 10.422811520890129 | 7.880952380952381 |

Original GP nominal 90% intervals cover **19/21** observed held-out measurements; observed mean nominal interval width **10.0970**. There is **no evidence that Gaussian process outperforms ridge**. The chapter's prior executed conditional generator produces two negative draws among 2,100 model-generated values, an explicit physical nonnegativity failure; generated values are **not measured original plant measurements**. The re-executed independent verification code is `qa/verify_brownlee_independent_loo.py` and the original full generation/figure code remains `code/section_6_28_industrial_stackloss_probabilistic_emulator/run.py`, with two scientifically meaningful observed-data figure generators and prior original output assets. R2 is satisfied by actual 20-in-training/1-held-out folds and no test leakage; R5 is satisfied by true independently computed Ridge and training-mean references; R6 reports errors/nominal interval coverage and physical sample violations. The certification covers **one historical plant, static retrospective conditional emulation only**. It does **not** certify a real-time digital twin, plant deployment or physical validity for all generated predictions.

## CH6-D02 — original real handwritten digit VAE–PCA

Original `sklearn.load_digits` has **1,797 actual scanned handwritten digits**, with a genuine, stratified disjoint **1,257 training / 270 validation / 270 test** split. The executed 35-epoch trained 2D VAE was replayed with recorded seed 19; the 2D PCA comparator was fitted on training images only. Original held-out test VAE reconstruction MSE **0.05387162044644356** and PCA MSE **0.05312391743063927**; paired PCA-minus-VAE MSE **−0.0007477067993022501**, bootstrap 95% interval **[−0.0015158850699663162, +0.000004024946656500377]**. The interval includes zero; **do not claim VAE superiority**. Genuine original digit reconstructions, latent space, validation curve and test-error plots were generated and inspected in earlier independent Pass 12/47/HEAD_REPLAY reports; exact checkpoint, predictions, figure and metrics hashes were independently matched to archived results in `figure_generation/vae_pca_results/REPLAY_CERTIFICATE_2026-11-12.md` and `HEAD_REPLAY_2026-11-12.md`. The complete executable code is `figure_generation/vae_pca_real_digits.py`. This Level A certification covers **one seeded held-out reconstruction demonstration**, not unconditional generation, cross-dataset performance or broad VAE superiority.

## CH6-D03 — original measured Mauna Loa CO₂ sequence modeling

The genuine Mauna Loa 1958–2001 atmospheric observations from `statsmodels.datasets.co2` were evaluated using the actually trained conditional VAE in `code/section_6_25_climate_co2_cvae/run.py` and original numerical verification in Pass 47–48. Original strict **chronological train=1,145, validation=254, test=306** overlapping forecast windows, 52-week observed conditioning history and eight-week forecast horizon; **three genuine independently trained random seeds** and **7,344 recorded test-prediction rows**. Source fit and original held-out forecasts were replayed by Pass 48, matching the original retained model metrics and predictions. Original CVAE prior-predictive mean **RMSE 0.4868 ppm**, Ridge **0.6673 ppm**, last value **1.4799 ppm**, seasonal drift **2.0913 ppm**. **Critical negative result:** nominal 90% prior-predictive bands cover only **20.25%**, meaning model uncertainty is severely miscalibrated. Additional later-period residual recalibration has observed **93.39%** coverage on 150 subsequent test windows, but this follow-up was **not integrated into the latest chapter** and overlapping windows invalidate distribution-free conformal coverage claims.

The scientific certificate covers the **truthfully reported empirical forecasting/error-and-calibration diagnostic**, **not a generally calibrated probabilistic forecast, full Earth System Model, meteorological safety system or physical climate simulator**. R6 is satisfied by explicitly **documenting failure** of a probabilistic calibration requirement, not by pretending the algorithm performed well. R7 offline domain validation applies to observed atmospheric time-series forecasting only. Model and source-data provenance, actual predictive outputs, baselines, plotted figures and execution records are in the source/Pass 48 package.

## R1–R10 proportional certification rationale

For all three specifically scoped studies, R1 real observed source provenance, R3 genuine model/algorithm execution, R4 real measured outputs and/or conditional generations distinguished from observed data, R8 genuine measured figures, R9 repository code/config/source traceability and R10 honest conclusions have been reviewed. R2 uses actual task-appropriate train/test or leave-one-out partitions; R5 genuinely computed reference methods, not rhetorical baselines; R6 error analysis appropriate to the exact experiment; and R7 real offline scientific observations, not mandatory physical deployment. This **does not retroactively waive missing test data or experiment execution for the parent named case studies**.

## Editorial and scientific separation

The user-provided **Pass 37 handoff note** states Figure 6.20 was placed on printed page 67, with end-of-chapter problems beginning on page 101, and a 109-page PDF/LaTeX package had been rebuilt. The actual Pass 37 file paths mentioned in that note **are not present in this runtime**; and Pass 45 is the later 109-page reported manuscript according to repo QA. Therefore the old note is only evidence of a prior placement assessment—not current final manuscript approval. No claim is made that Figure 6.20 source/caption mapping has been rechecked in the latest compiled chapter. The figure number 6.20 also appeared in a separate digit VAE figure-generation script and requires actual manuscript/asset verification before final release.

**Decision:** three new, explicitly scoped valid Level A scientific certificates; fourteen parent full Chapter 6 case studies/applications remain governed by their separate evidence matrix and unresolved broad claims; Chapter 6 production HOLD.
