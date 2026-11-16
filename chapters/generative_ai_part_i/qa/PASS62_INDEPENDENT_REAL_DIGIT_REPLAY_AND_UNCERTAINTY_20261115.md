# Chapter 6 — Pass 62: independently replayed original-data evaluator control

**Date:** 2026-11-15  
**Master QA rule:** proportional R1–R10; original observed data, actual model code/outputs and source-derived evidence; separate scoped science from production approval.

## Actual execution performed

Opened the retained `Chapter6_Pass61_Real_Classifier_Control.zip`, which contains the exact prior Python real-digit evaluator experiment, original `figure_6_44_class_counts_pass27.csv`, measured results and image outputs. Reran the archived `audit_digit_classifier_control.py` using its original class-count source, against real `sklearn.datasets.load_digits` data with seed **20261109**. Original held-out stratified partition reproduced **1257 training / 270 validation / 270 independent test** real-image samples, with the exact test-index-array SHA-256 `d945a001641ccbaecf31384feb1d1ba6e54c4d0dc33a3f6c9a40effe505514bb`.

The independent replay reproduced:
- **257/270** correctly recognized original real held-out digits (**95.185185185%**);
- all **10** original-image class-specific records, exact numeric content;
- all **100 cells** of the real held-out 10×10 confusion matrix, exact numeric content;
- saved prior model-output class-count source checksum `1b3499a3fa7f93e0cd044ade2b827858393d6ad54c2bedf9cb99a9c7916b2fb0`;
- saved previously generated class counts **778/1000** conditional classifier-label agreements, **94/1000** unconditional assigned-comparison-label matches.

The rerun reproduced original metrics and tables; PDF/manuscript integration was **not** inspected in this execution. **The original latent diffusion generator was not retrained**. Original class agreement is an earlier executed generator result, not an original observation.

## Scientific R6 uncertainty remediation

A source-based Wilson 95% sample proportion interval calculation was executed for real classifier test accuracy and for saved conditional generated-output label agreement. The full independent source and resulting per-digit intervals are archived in the user-facing Pass 62 ZIP. The matching reusable book code and source-derived JSON numerical output were added to the section's GitHub folder:

- `code/section_6_20_1_latent_diffusion_digits/compute_sample_uncertainty.py`;
- `code/section_6_20_1_latent_diffusion_digits/original_evaluator_control/sample_uncertainty.json`.

| Quantity | Observed diagnostic fraction | 95% Wilson interval |
|---|---:|---:|
| Original real held-out digit classification accuracy | 257/270 = 95.19% | 91.94%–97.16% |
| Prior saved conditional-generated label agreement | 778/1000 = 77.8% | 75.12%–80.27% |

**Limitations are essential:** These are distinct groups of images (not paired). The generated samples are outputs of **one fitted generator**, may be dependent, and these sample-level intervals do **not** quantify training-seed variability or calibrated generator generality. A classifier's requested-class agreement is not a validated human visual-quality metric. No statistical test treating the two groups as random independent observations from a common distribution was performed.

The GitHub Actions workflow `.github/workflows/chapter6_scoped_level_a_certification.yml` was updated to compute the uncertainty evidence and check its key source totals. A newly completed GitHub Actions job has **not** been independently observed.

## Source integrity and archived package

The user-facing `Chapter6_Pass62_Independent_Scientific_Replay.zip` contains original data-summary counts, archived execution script, original versus independently rerun outputs, uncertainty code and numerical JSON, execution log, and a SHA-256 file manifest. The local rerun uses the archived executed Pass 61 script; the separate current GitHub code `audit_real_evaluator_control.py` has substantively equivalent experimental classifier/split logic and its own committed results. Do not conflate bytewise source identity across those two script names.

## Production audit boundary

Actual GitHub root `chapters/generative_ai_part_i/` still exposes **no current full `manuscript/` LaTeX and PDF release**. The user-provided `Proceed Chapter 6.txt` reports that Figure 6.20 was corrected and visually verified on printed page 67 in a prior 109-page manuscript; **its text is not the current binary PDF, source image or direct source-to-caption figure mapping**. The production figure-manifest still marks Figure 6.20 source provenance unverified.

The current source register has **12 narrowly scoped scientific certificates** (11 Level A, 1 Level B). They are **not** 12 certified parent applications. The last audited chapter baseline includes **14 parent empirical applications, 48 figures, 17 tables and 11 numbered listings**; present manuscript integration cannot be directly certified from archived counts. Driving perception-benefit claims need a genuine multi-scene held-out downstream evaluation, and physical performance claims for newly generated concrete formulations require actual independent specimen tests.

**Pass 62 determination:** Independent original real-data classifier replay **PASS**; source-derived descriptive uncertainty **PASS**; new code/data output committed to GitHub. Existing D12 *scoped* certificate receives stronger support but no promotion to Stable Diffusion or broad natural-image scientific certification. Full manuscript and parent-application production certification remain **HOLD**.
