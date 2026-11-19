# Chapter 6 — Pass 67: authentic NEU matched-seed effect and real digit-model run reconciliation

**Scientific audit date:** 2026-11-18.  
**Governing standard:** proportional book-wide Master QA R1–R10. Original observed data, actual executed model outputs, scientific analysis, complete repository code, source-linked figures, scoped truth and separate manuscript production acceptance.

## 1. New original-data scientific verification of CH6-D04 (Level B negative result)

**Evidence:** the actual committed `code/section_8_8_2_neu_defect_benchmark/replicated_seed_metrics_pass13.csv`, original observed source-result SHA-256 `1f74ab0c8067612a072c2eff5554b971682ce851bd4c7ca5eb585fbb5f163d9e` (Git blob `56de24a4f74f2c87e0d01b3b97c811e00b317f75`). Exactly **12 executed classifier training-arm/seed score rows**, 3 original training seeds (20261106, 20261107, 20261108) and 4 arms. These are predictions scored on the **same 360 original photographed steel defects** in each seed. They are not 4,320 independent photographs.

The input CSV passed an independent original-source checksum and full seed/arm key validation. Original measurements were independently recomputed with Python `statistics`, `scipy.stats.t` and `scipy.stats.binomtest`, distinct from the new standard-library repository implementation.

| Genuine observed paired effect: GAN minus reference | Mean (percentage points) | Three paired seed effects (percentage points) | Descriptive Student-t 95% interval (pp) |
|---|---:|---|---|
| Accuracy minus original real-only | **−19.2593** | −18.6111, −19.4444, −19.7222 | −20.6957 to −17.8228 |
| Macro-F1 minus original real-only | **−20.6174** | −20.6921, −22.1786, −18.9814 | −24.5918 to −16.6429 |
| Accuracy minus matched real-image repetitions/flips | **−26.9444** | −24.7222, −22.5000, −33.6111 | −41.5498 to −12.3391 |
| Macro-F1 minus matched real-image repetitions/flips | **−28.7086** | −27.1225, −24.7150, −34.2882 | −41.0791 to −16.3381 |

**Crucial Master QA interpretation:** All three matched training-seed effects are negative for each metric/comparator, supporting a negative finding **within this original, short-budget NEU benchmark**. But with only three fitted seeds, the **exact one-sided sign-test tail is p = 0.125** (3/3 negative); it does not reject a symmetric no-direction null at 0.05. The Student-t intervals with **two degrees of freedom** are descriptive and strongly assumption-sensitive; they must not be represented as factory/domain generalization guarantees. The per-image paired predictions are not in this aggregate CSV; do not present a McNemar or 360-independent-trials significance test. The study does not certify GAN visual defect fidelity, external plants, different training budgets or deployment.

### Actual added executable scientific code and results
- `code/section_8_8_2_neu_defect_benchmark/audit_paired_seed_uncertainty.py`: complete original-source validation, paired accuracy/F1 effects, three-seed variability, exact three-pair sign probabilities and deterministic scientific SVG generation.
- `code/section_8_8_2_neu_defect_benchmark/verify_neu_paired_independent_scipy.py`: independent SciPy original-source calculation checking the same effects, source SHA and statistical limits.
- `code/section_8_8_2_neu_defect_benchmark/test_neu_paired_seed_uncertainty.py`: seven software fixture/source-integrity tests (real 12-row seed check; corrupted numeric input; duplicate, absent or unregistered seeds; impossible scores).
- `code/section_8_8_2_neu_defect_benchmark/paired_seed_science/{paired_seed_statistics.json,paired_seed_effects.csv,paired_seed_effects.svg}`: retained genuine source-derived numeric effects and annotated vector scientific figure.
- `code/section_8_8_2_neu_defect_benchmark/plot_neu_paired_seed_evidence.py`: additional readable scientific one-panel measured-pair plot generator; the local compiled plot was visually inspected, and an initially clipped headline was fixed.
- Original measured CSV and `README.md` remain in the correct section folder.

**Production figure provenance verification:** The exact normalized SVG from a locally executed source-equivalent Python `draw_svg` reproduction yielded Git blob SHA-1 **`4e5fa422a4066652ad240ee0797be3f953de2042`**, which matched the resulting actual committed GitHub SVG blob byte for byte. The authored Python source now uses deterministic 0.001-unit SVG numeric formatting to prevent float-representation drift. CI was extended to re-run the original analysis, compare its numerical results against the committed record and independently validated SciPy results, and reject an SVG byte mismatch. **Completed new GitHub Actions CI status was not independently observed during this increment**; the independent local SciPy calculation was executed.

The CH6-D04 scoped scientific register entry was remediated to record the actual paired effect and honest n=3 inference; its **existing Level B SCIENTIFIC_CERTIFIED** decision remains limited to this offline original-data negative experiment. This is not a newly approved parent manuscript case study.

## 2. Reconcile actual CH6-D12 digit-generator experiments without falsifying a rerun

The archived prior real-model source has **778/1000** conditional classifier-label matches and **94/1000** unconditional assigned-reference-label matches. A later genuinely executed GitHub Actions original scanned-digit run **44209629586** produced **779/1000** conditional, **94/1000** unconditional. Those are different stochastic model executions; they need not produce identical binaries or exact classifier scores.

Both retained original 22-row per-class source files were independently parsed; every denominator is 100 generated outputs per digit. The **only per-class difference** is conditional digit **8: 66/100 in the archive, 67/100 in the newer executed run**, explaining the overall difference of one. The prior archived source SHA-256 is `1b3499a3fa7f93e0cd044ade2b827858393d6ad54c2bedf9cb99a9c7916b2fb0`; the current run summary SHA-256 is `5dbc753223f563146a1d11cbcb2850940ec0a88f11bfdb3c69a6287889068115`.

Added `code/section_6_20_1_latent_diffusion_digits/verify_independent_digit_run_difference.py` and a committed run-specific `qa/ci_44209629586/archived_vs_current_digit_science.json`. `qa/validate_scoped_ch6_certifications.py` now checks the two archived/current actual results **separately**, instead of rejecting 779 as an invented deviation from 778. The CH6-D12 evidence-limit record has also been updated to preserve both executions. This strengthens reporting integrity, not the application-level breadth of natural-language Stable Diffusion.

## 3. Production and final Master QA gate

The last known compiled Chapter 6 benchmark inventory (109 pages, 48 figures, 17 tables, 11 listings, 14 parent application/case-study units) is **historical**. The current exact compiled PDF, LaTeX and 14 full parent titles/outputs are still absent from the accessible canonical Chapter 6 root. The user's previous Pass37 handoff states Figure 6.20 was placed within the industrial case study on page 67, but the actual image, caption, generator identity and final current PDF are **not directly available**.

The existing 12 **limited** scientific certificates (11 A and 1 B) must not be promoted to 12 of the 14 parent applications. The release gate requires distinct manuscript-titled B/C parent R1–R10 evidence and actual source-linked figures, tables, listings and exact current LaTeX/PDF. Synthetic-driving claims still require representative real multi-scene driving and independently assessed downstream perception; independent physical strength for generated concrete proposals still needs actual cast/test observations.

**Decision:** CH6-D04 additional real-data scientific uncertainty audit **PASS within its narrow observed scope**, independent new vs archived digit generation result reconciliation **PASS**, source-linked SVG **byte-identically reconciled**, all source/verification/figure-generation code committed to GitHub. No new parent application certified. **Full Chapter 6 scientific and production release remains HOLD** until actual parent-task evidence and manuscript inspection establish Master QA compliance.
