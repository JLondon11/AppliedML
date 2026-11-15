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
