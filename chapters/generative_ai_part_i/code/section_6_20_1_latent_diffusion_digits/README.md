# Chapter 6 §6.20.1 — Class-conditioned latent diffusion on real digit images

**Scope:** Executed VAE latent diffusion model with a class-label embedding (0–9). **Not** pretrained Stable Diffusion, natural-language text-to-image, cross-attention, or classifier-free guidance.

Dataset: genuine handwritten digit images distributed by `sklearn.datasets.load_digits()`; 1,797 images of size 8×8, stratified split 1,257 training / 270 validation / 270 untouched test. Fixed seed 20261109. Sixteen-dimensional VAE, 25 epochs (validation checkpoint), 48-step beta schedule, 35 epochs per diffusion variant, and 1,000 sampled images per variant. Independent logistic evaluator trains **only** on original training images, achieving 95.19% accuracy on untouched genuine digit test images.

| Evaluation | Conditional | Unconditional |
|---|---:|---:|
| Generated sample matches requested digit by real-trained classifier | 77.8% | 9.4% |
| Mean closest real training-image pixel MSE | 0.01980 | 0.02337 |

Held-out VAE reconstruction RMSE: 0.162. All images, classifier probabilities, two diffusion checkpoints, VAE checkpoint, split indices, model training histories, manifest, figures, and verified source script are in the Pass 23 chapter ZIP.

Install `torch`, `numpy`, `scikit-learn`, `matplotlib`; run `python run.py --vae-epochs 25 --diffusion-epochs 35` then `python test_outputs.py`. The generated NumPy-array checkpoint artifacts must only be opened from trusted local runs; the verification script uses `weights_only=False` for those generated NumPy objects.

**Limitations:** one model seed; low-resolution monochrome digits; label embeddings rather than natural language; no trained pixel-space diffusion control; classifier evaluation is only an indirect measure of semantic quality. The manuscript Stable Diffusion case study has **partial real-data evidence** and remains **HOLD** for complete scientific certification.

Chapter manuscript insertion: Figure 6.43, Figure 6.44 and Table 6.25 in the Pass 23 edition. Full data and generated results are in the downloadable chapter ZIP, not committed as binary tensors to GitHub.

## Scoped scientific certification — 2026-11-14

The **actual label-conditioned latent diffusion diagnostic on 8×8 original scanned handwritten digits** is registered as **CH6-D12, Level A SCIENTIFIC_CERTIFIED** for this limited class-conditioned generator and its original real-digit-trained classifier evaluation. Original-source sklearn digits (1,797 genuine scans; 1257/270/270 train/validation/test), trained VAE and two diffusion variants, 1,000 actual generated outputs per arm, 2,000 individual saved probability rows, original model checkpoints and authentic model-output figures were verified in earlier Pass23/47 reports. The classifier agrees with requested label on **778/1000** conditional samples and **94/1000** unconditional samples. These are not independent externally labeled real-image generalization measurements; this is an indirect diagnostic of actual generated digits and one model-training seed. Source class summary and its per-class totals are mechanically reconciled by `qa/validate_scoped_ch6_certifications.py`.

**Do not state that this experiment is Stable Diffusion, text-prompt generation, classifier-free guidance, natural-image text-to-image synthesis, or full held-out generative-fidelity certification.** This certificate does not promote that broader chapter parent application, and final compiled manuscript figure placement is separately **PRODUCTION_HOLD**.

## Class-by-class evidence and publication figure (Pass 60)

The original held-out-real-trained digit classifier agrees with the conditioning label for **778/1,000 conditional** outputs versus **94/1,000 unconditional** outputs assigned comparison labels. These are *model-output diagnostic agreements*, not externally observed image-quality grades or independent new test digit photographs. The original class-level source file `figure_6_44_class_counts_pass27.csv` shows conditional agreement from **39/100** (digit 1) to **99/100** (digit 4); digit 2 has **55/100**, digit 5 **67/100** and digit 8 **66/100**. Reporting only the 77.8% overall rate conceals meaningful class heterogeneity. One fitted generator seed is insufficient to establish cross-seed generator robustness.

A new two-panel scientific SVG `figures/verified_digit_conditioning_science.svg` is generated deterministically by `node plot_verified_conditioning_pipeline.js`. Panel A displays **actual original scored per-class outputs** using paired class dots, rather than generic bars. Panel B accurately diagrams the executed real-scanned-digit / split / 16-dimensional VAE / 48-step latent diffusion / independent real-trained classifier pipeline. The schematic depicts the experimental method, not invented physical measurements. Both source and generated SVG are in this exact section's repository folder, and CI will fail if source-data regeneration changes the committed SVG. **Manuscript figure numbering and actual PDF placement remain separately unverified**; do not silently replace or relabel any existing published figure without checking current LaTeX.

## Independent genuine held-out evaluator control — Pass 61

The actual experimental evaluator in `run.py` was independently refitted on genuine source `sklearn.datasets.load_digits` with its original seed **20261109** and exact stratified split (**1,257 training / 270 validation / 270 untouched test images**). Its original-real test classifier accuracy is **257/270 = 95.1852%**, matching the earlier original model's reported aggregate. Ten-class held-out confusion counts and original-data class recall were recomputed and preserved.

Compare this **original real-image classifier control** with earlier **saved generated-image classifier agreement**, without asserting the two groups are paired samples: digit 1 real-test recall **25/27 (92.59%)** versus conditional generated label agreement **39/100 (39%)**; digit 2 real **26/27 (96.30%)** versus generated **55/100 (55%)**; digit 8 real **25/26 (96.15%)** versus generated **66/100 (66%)**. The observed severe generated-sample deficit for these classes is not explained by failure to recognize most genuine real examples. Digit 4 shows the reverse descriptive ordering (real **24/27**, generated **99/100**), which also cautions against treating cross-distribution differences as a model comparison or human realism measure.

- `audit_real_evaluator_control.py`: complete real-data refitting experiment, numerical evidence, confusion matrix and scientific plot generation.
- `original_evaluator_control/{real_evaluator_vs_generated_class.csv,real_evaluator_confusion.csv,metrics.json}`: committed genuine measured held-out classifier scores and original saved generated-output comparisons.
- `original_evaluator_control/real_classifier_control_summary.svg`: source-linked two-panel vector graphic (real-vs-generated class dots and measured real-classifier confusion) generated by `plot_real_evaluator_control.js`.
- `verify_original_evaluator_control.py`: fail-closed source-row, per-class, total and held-out checksum consistency checks.
- The GitHub workflow now independently refits the original-real evaluator on CI, verifies numeric agreement, and checks SVG regeneration drift.

**Scope:** This evaluates the classifier as a control, not generator diversity, perceptual fidelity, test-set independence between generated outputs, deployment, Stable Diffusion, or cross-generator-seed uncertainty. The original source counts and real classifier results have different denominators and are not paired. Figures are source-linked publication candidates; their actual placement in the current compiled Chapter 6 cannot yet be verified.

## Pass 62 independent replay and scoped uncertainty

The archived Pass 61 real-original-digit evaluator was independently re-executed in Pass 62 on the actual `sklearn.datasets.load_digits` source and original split. All ten measured per-class rows, the original real held-out 10×10 confusion counts, original 257/270 test accuracy and source/test-index hashes numerically matched the previously saved records. The generator itself was **not** retrained. Details: `../../qa/PASS62_INDEPENDENT_REAL_DIGIT_REPLAY_AND_UNCERTAINTY_20261115.md`.

`compute_sample_uncertainty.py` calculates transparent, descriptive sample Wilson intervals from the observed evaluator and saved class-count records; its numerical results are at `original_evaluator_control/sample_uncertainty.json`. The 95% interval for genuine held-out real-classifier accuracy is **[91.94%, 97.16%]**, and for saved conditional-generator classifier agreement is **[75.12%, 80.27%]**. **These are not paired groups**, and the generated-sample interval is conditional on a single trained generator; it does not establish between-generator-seed repeatability or subjective image quality. The continued Level A certification covers the narrow actual original-digit task only.

## Pass 66 — independent executed-model original-image source gallery and publication code (2026-11-18)

The actual original-data Chapter 6 digit experiment now has **two print-width, image-first scientific output plates**, not just plotted per-class accuracy. They are generated from the original true held-out 8×8 sklearn source scans and **actual saved model-generated pixel tensors** from completed GitHub Actions run **44209629586**. For each digit class 0–9 the figure displays the first source **real held-out image**, the first genuinely generated **class-conditioned** image and the first genuinely generated **unconditional** image, with the actual classifier prediction and whether it matches the requested label. Selection is fixed by source position rather than image aesthetics or label success; failures are deliberately **not hidden**.

The **new executed CI run** (different from the previous archived Pass27 run) observed **779/1000** conditional and **94/1000** unconditional label matches. The older **778/1000** conditional archive remains a valid previous run, not a contradictory claim. Observed conditional counts for digit classes 0–9 in the new run are `93,39,55,92,99,67,92,91,67,84`; class 1 is notably weak at 39/100. New genuine-heldout-scans classifier accuracy for the latest run was **258/270 = 95.56%**, versus the separately executed earlier classifier-control 257/270 reported above. That one-image difference is **not** an independent statistical experiment or a claim of a model architecture upgrade.

**Actual complete experiment evidence and production references:**

- Real digit generator run: [GitHub Actions 44209629586](https://github.com/JLondon11/AppliedML/actions/runs/44209629586).
- Genuine image-first figure source: `render_actual_generated_vs_original_digits.py`.
- Independent 2000-original-output and 30-image source-pixel audit: `../../qa/verify_pass66_actual_digit_image_evidence.py`.
- Committed original-data and generated-model figure PDFs, PNGs, SVGs, exact per-build hashes and source-provenance JSON: `../../figure_assets/verified_scoped_experiments/CH6_D12_real_digit_diffusion/`.
- Figure binary commit workflow [run 44424877916](https://github.com/JLondon11/AppliedML/actions/runs/44424877916), **SUCCESS**, with two independently visually inspected 6.5-inch vector PDFs, 15 input/output images per page, minimum font **8.1 pt** and zero clipped text. Original successful figure-build run **44423296969** has identical rendered pixels despite normal varying PDF creation metadata.
- Formal claim-restricted scientific and production report: `../../qa/PASS66_EXECUTED_DIGIT_IMAGE_SCIENCE_CERTIFICATION_20261118.md`.

**Important scientific limits:** All generated images are numerical learned-model outputs, not newly photographed observations. The test photos are genuine real observations, the two generator arms share a single trained-seed setting, and classifier requested-label agreement is an *indirect* diagnostic rather than expert-validated image realism. This is **not** natural-language text-to-image Stable Diffusion, and neither a new clinical task nor independent generator-seed generalization is established.

**Scientific status:** existing **CH6-D12 Level A SCIENTIFIC_CERTIFIED** limited to the actual real-scanned-digit evaluation; **no additional parent application** or experiment certificate is implied. **Production status:** standalone figures print-width staged and **actually committed** in the repository, but **current complete Chapter 6 PDF/TeX insertion, numbering, citations and captions are not verified**; overall chapter release remains **HOLD**.
